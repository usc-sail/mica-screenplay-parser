# author : Sabyasachee

# standard library imports
import math
from multiprocessing import Pool, current_process
import os
import sys

# third party imports
import numpy as np
import pandas as pd
import torch
from torch.optim import Adam

# user imports
from movieparser.scriptloader import get_dataloaders, label2id
from movieparser.scriptparser import ScriptParser
from movieparser.evaluate import evaluate, evaluate_movie

def train_movie(movie: str, device:torch.device, results_folder: str, seqlen: int, train_batch_size: int, \
    eval_batch_size: int, leave_one_movie_out: bool, encoder_learning_rate: float, learning_rate: float, \
    max_epochs: int, max_norm: int, verbose = False) -> pd.DataFrame:
    
    name = current_process().name
    if movie == "-":
        print("{}: training on (movie, error) stratified splits".format(name))
    else:
        print("{}: training on all movies except {}".format(name, movie))

    train_loader, test_loader, dev_loader = get_dataloaders(results_folder, seqlen, train_batch_size, \
        eval_batch_size, movie, leave_one_movie_out, device)
    print("{}: {} train batches, {} dev batches, {} test batches".format(name, len(train_loader), len(dev_loader), \
        len(test_loader)))

    feature_size = train_loader.features.shape[2]
    scriptparser = ScriptParser(feature_size, len(label2id))
    scriptparser.to(device)

    optimizer = Adam([
        {"params": scriptparser.encoder.parameters(), "lr": encoder_learning_rate},
        {"params": [parameter for name, parameter in scriptparser.named_parameters() if not name.startswith("encoder")]}
    ], lr=learning_rate)

    for epoch in range(max_epochs):
        if verbose:
            print("{}: epoch {:2d}".format(name, epoch + 1))
        train_losses = []
        scriptparser.train()

        for b, (train_scripts, train_features, train_labels) in enumerate(train_loader):
            loss, _ = scriptparser(train_scripts, train_features, train_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(scriptparser.parameters(), max_norm=max_norm)
            optimizer.step()
            optimizer.zero_grad()
            train_losses.append(loss.detach().cpu().item())
            if verbose and (b + 1) % 50 == 0:
                print("{}: \tbatch {:3d} avg. train loss = {:.3f}".format(name, b + 1, np.mean(train_losses)))

        dev_perf, dev_loss = evaluate(scriptparser, dev_loader)
        if verbose:
            print("{}: dev evaluation:".format(name))
            print(dev_perf)
            print()
        print("{}: epoch {:2d} train loss = {:.3f}, dev loss = {:.3f}".format(name, epoch + 1, np.mean(train_losses), \
            dev_loss))

    test_perf, _ = evaluate(scriptparser, test_loader)
    if verbose:
        print("{}: test evaluation:".format(name))
        print(test_perf)
        print()
    print("{}: F1 S={:.3f}, N={:.3f}, C={:.3f}, D={:.3f}, T={:.3f}, E={:.3f}, M={:.3f}".format(name, \
        test_perf.loc["S", "f1"], test_perf.loc["N", "f1"], test_perf.loc["C", "f1"], test_perf.loc["D", "f1"], \
        test_perf.loc["T", "f1"], test_perf.loc["E", "f1"], test_perf.loc["M", "f1"]))

    return test_perf

def train(data_folder: str, results_folder: str, seqlen: int, train_batch_size: int, eval_batch_size: int, \
    eval_movie: str, leave_one_movie_out: bool, learning_rate: float, encoder_learning_rate: float, \
    max_epochs: int, patience: int, max_norm: float, parallel: bool, n_folds_per_gpu: int):
    
    movies = [line.split()[0] for line in \
        open(os.path.join(data_folder, "SAIL_annotation_screenplays/line_indices.txt")).read().splitlines()]
    
    if leave_one_movie_out:
        if eval_movie in movies:
            movies = [eval_movie]
        elif eval_movie == "all":
            pass
        else:
            print("movie not found")
            sys.exit(-1)
    else:
        movies = ["-"]

    n_gpus = torch.cuda.device_count()
    n_processes = n_gpus * n_folds_per_gpu
    n_iterations = math.ceil(len(movies)/n_processes)
    arguments = [results_folder, seqlen, train_batch_size, eval_batch_size, leave_one_movie_out, \
        encoder_learning_rate, learning_rate, max_epochs, max_norm]
    test_perfs = []

    print("starting multiprocess training:\n\n")

    for i in range(n_iterations):
        process_movies = movies[i * n_processes: (i + 1) * n_processes]
        process_devices = [torch.device(j % n_gpus) for j in range(n_processes)]
        process_arguments = [tuple([process_movies[j], process_devices[j]] + arguments) for j in range(n_processes)]
        
        with Pool() as pool:
            process_test_perfs = pool.starmap(train_movie, process_arguments)
        test_perfs.extend(process_test_perfs)
    print("\n\nmultiprocess training ends")
    
    test_perf = sum(test_perfs)
    confusion = test_perf.iloc[:, :test_perf.shape[0]]
    test_perf["precision"] = (confusion/confusion.sum(axis=0)).values.diagonal()
    test_perf["recall"] = (confusion/confusion.sum(axis=1)).values.diagonal()
    test_perf["f1"] = 2 * test_perf["precision"] * test_perf["recall"] / (test_perf["precision"] + test_perf["recall"])

    print()
    print("overall test evaluation:")
    print(test_perf)