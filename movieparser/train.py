# author : Sabyasachee

# standard library imports
import math
import os
import sys
import time

# third party imports
import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch.multiprocessing import Pool, set_start_method, current_process

# user imports
from movieparser.scriptloader import get_dataloaders, label2id
from movieparser.scriptparser import ScriptParser
from movieparser.evaluate import evaluate, evaluate_movie

def train_movie(movie: str, device:torch.device, results_folder: str, seqlen: int, bidirectional: bool, train_batch_size: int, eval_batch_size: int, encoder_learning_rate: float, learning_rate: float, max_epochs: int, max_norm: float, verbose = False) -> pd.DataFrame:
    
    name = current_process().name
    if movie == "-":
        print("{:25s}: training on (movie, error) stratified splits".format(name))
    else:
        print("{:25s}: training on all movies except {}".format(name, movie))

    train_loader, test_loader, dev_loader = get_dataloaders(results_folder, seqlen, train_batch_size, eval_batch_size, movie, device)
    print("{:25s}: {:3d} train batches, {:3d} dev batches, {:3d} test batches".format(name, len(train_loader), len(dev_loader), len(test_loader)))

    feature_size = train_loader.features.shape[2]
    scriptparser = ScriptParser(feature_size, len(label2id), bidirectional)
    scriptparser.to(device)

    optimizer = Adam([
        {"params": scriptparser.encoder.parameters(), "lr": encoder_learning_rate},
        {"params": [parameter for name, parameter in scriptparser.named_parameters() if not name.startswith("encoder")]}
    ], lr=learning_rate)

    for epoch in range(max_epochs):
        if verbose:
            print("{:25s}: epoch {:2d}".format(name, epoch + 1))
        train_losses = []
        scriptparser.train()
        start = int(time.time())

        for b, (train_scripts, train_features, train_labels) in enumerate(train_loader):
            loss, _ = scriptparser(train_scripts, train_features, train_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(scriptparser.parameters(), max_norm=max_norm)
            optimizer.step()
            optimizer.zero_grad()
            train_losses.append(loss.detach().cpu().item())
            if (b + 1) % 20 == 0:
                print("{:25s}: epoch {:2d} batch {:3d} avg. train loss = {:.3f}".format(name, epoch + 1, b + 1, \
                    np.mean(train_losses)))

        dev_perf, dev_loss = evaluate(scriptparser, dev_loader)
        if verbose:
            print("{:25s}: dev evaluation:".format(name))
            print(dev_perf)
        delta = int(time.time()) - start
        minutes, seconds = delta//60, delta%60
        print("{:25s}: epoch {:2d} avg. train loss = {:.3f}, dev loss = {:.3f}, F1 S={:.3f} N={:.3f} C={:.3f} D={:.3f} T={:.3f} E={:.3f} O={:.3f}, macro={:.3f} micro={:.3f} weighted={:.3f}, time taken = {:2d} min {:2d} sec".format(name, epoch + 1, np.mean(train_losses), dev_loss, dev_perf.loc["S", "f1"], dev_perf.loc["N", "f1"], dev_perf.loc["C", "f1"], dev_perf.loc["D", "f1"], dev_perf.loc["T", "f1"], dev_perf.loc["E", "f1"], dev_perf.loc["O", "f1"], dev_perf["macro-f1"].values[0], dev_perf["micro-f1"].values[0], dev_perf["weighted-f1"].values[0], minutes, seconds))

    test_perf, _ = evaluate(scriptparser, test_loader)
    if verbose:
        print("{:25s}: test evaluation:".format(name))
        print(test_perf)
        print()
    print("{:25s}: F1 S={:.3f}, N={:.3f}, C={:.3f}, D={:.3f}, T={:.3f}, E={:.3f}, O={:.3f}, macro={:.3f} micro={:.3f} weighted={:.3f}".format(name, test_perf.loc["S", "f1"], test_perf.loc["N", "f1"], test_perf.loc["C", "f1"], test_perf.loc["D", "f1"], test_perf.loc["T", "f1"], test_perf.loc["E", "f1"], test_perf.loc["O", "f1"], test_perf["macro-f1"].values[0], test_perf["micro-f1"].values[0], test_perf["weighted-f1"].values[0]))

    return test_perf

def train(data_folder: str, results_folder: str, seqlen: int, bidirectional: bool, train_batch_size: int, eval_batch_size: int, eval_movie: str, leave_one_movie_out: bool, learning_rate: float, encoder_learning_rate: float, max_epochs: int, patience: int, max_norm: float, parallel: bool, n_folds_per_gpu: int, verbose = False):
    
    movies = [line.split()[0] for line in open(os.path.join(data_folder, "SAIL_annotation_screenplays/line_indices.txt")).read().splitlines()]
    
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

    if parallel:
        n_gpus = torch.cuda.device_count()
        n_processes = n_gpus * n_folds_per_gpu
    else:
        n_gpus = 1
        n_processes = 1
    
    n_iterations = math.ceil(len(movies)/n_processes)
    arguments = [results_folder, seqlen, bidirectional, train_batch_size, eval_batch_size, encoder_learning_rate, learning_rate, max_epochs, max_norm, verbose]
    test_perfs = []

    print("starting multiprocess training:\n")
    set_start_method('spawn', force=True)

    for i in range(n_iterations):
        process_movies = movies[i * n_processes: (i + 1) * n_processes]
        process_devices = [torch.device(j % n_gpus) for j in range(len(process_movies))]
        process_arguments = [tuple([process_movies[j], process_devices[j]] + arguments) for j in range(len(process_movies))]
        
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