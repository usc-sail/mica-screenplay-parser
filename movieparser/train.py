from movie_screenplay_parser.movieparser.scriptloader import get_dataloaders, label2id
from movie_screenplay_parser.movieparser.scriptparser import ScriptParser
from movie_screenplay_parser.movieparser.evaluate import evaluate, get_classification_report

import json
import math
import os
import sys
import time
from typing import Dict, Any, List
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import torch
from torch.optim import Adam
from torch.multiprocessing import Pool, set_start_method, current_process

def train_movie(movie: str, device: torch.device, results_folder: str, seqlen: int, bidirectional: bool, train_batch_size: int, segment_lengths: List[int], encoder_learning_rate: float, learning_rate: float, max_epochs: int, max_norm: float, verbose = False) -> Dict[str, Any]:
    
    #####################################################################
    #### get name
    #####################################################################

    name = current_process().name
    if movie == "-":
        print("{:25s}: training on (movie, error) stratified splits".format(name))
    else:
        print("{:25s}: training on all movies except {}".format(name, movie))

    #####################################################################
    #### get train and test loader
    #####################################################################

    train_loader, test_loader = get_dataloaders(results_folder, seqlen, train_batch_size, movie, device)
    print("{:25s}: {:3d} train batches, {:3d} test batches".format(name, len(train_loader), len(test_loader)))

    #####################################################################
    #### initialize model
    #####################################################################
    
    feature_size = train_loader.features.shape[2]
    scriptparser = ScriptParser(feature_size, len(label2id), bidirectional, device_index=device.index)
    scriptparser.to(device)
    print("{:25s}: scriptparser model loaded onto GPU {}".format(name, device.index))

    #####################################################################
    #### initialize optimizer
    #####################################################################
    
    optimizer = Adam([
        {"params": scriptparser.encoder.parameters(), "lr": encoder_learning_rate},
        {"params": [parameter for name, parameter in scriptparser.named_parameters() if not name.startswith("encoder")]}
    ], lr=learning_rate)

    #####################################################################
    #### initialize output dictionary
    #####################################################################
    
    output_dict = {}

    #####################################################################
    #### start training
    #####################################################################
    
    for epoch in range(max_epochs):

        #####################################################################
        #### print epoch number, set model to training, and start timer
        #####################################################################
        
        if verbose:
            print("{:25s}: epoch {:2d}".format(name, epoch + 1))
        train_losses = []
        scriptparser.train()
        start = int(time.time())

        #####################################################################
        #### train batch
        #####################################################################
        
        for b, (train_scripts, train_features, train_labels) in enumerate(train_loader):
            loss, _ = scriptparser(train_scripts, train_features, train_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(scriptparser.parameters(), max_norm=max_norm)
            optimizer.step()
            optimizer.zero_grad()
            train_losses.append(loss.detach().cpu().item())
            if (b + 1) % 20 == 0:
                print("{:25s}: epoch {:2d} batch {:3d} avg. train loss = {:.3f}".format(name, epoch + 1, b + 1, np.mean(train_losses)))

        #####################################################################
        #### evaluate test scripts
        #####################################################################
        
        eval_dict = evaluate(scriptparser, test_loader, name, segment_lengths)
        output_dict["epoch_{}".format(epoch + 1)] = eval_dict
        
        #####################################################################
        #### find elapsed time
        #####################################################################
        
        delta = int(time.time()) - start
        minutes, seconds = delta//60, delta%60
        
        #####################################################################
        #### print train loss, elapsed time, and test micro scores
        #####################################################################
        
        print("{:25s}: epoch {:2d} avg. train loss = {:.3f}, time taken = {:2d} min {:2d} sec".format(name, epoch + 1, np.mean(train_losses), minutes, seconds))
        for segment_length, segment_dict in eval_dict.items():
            print("{:25s}: epoch {:2d} segment len = {:3d} micro P = {:.3f} R = {:.3f} F1 = {:.3f}".format(name, epoch + 1, segment_length, segment_dict["micro"]["precision"], segment_dict["micro"]["recall"], segment_dict["micro"]["f1"]))

    return output_dict





def train(data_folder: str, results_folder: str, seqlen: int, bidirectional: bool, train_batch_size: int, eval_movie: str, learning_rate: float, encoder_learning_rate: float, max_epochs: int, max_norm: float, parallel: bool, n_folds_per_gpu: int, segment_lengths: List[int], verbose = False):
    
    #####################################################################
    #### get list of movies
    #####################################################################
    
    movies = [line.split()[0] for line in open(os.path.join(data_folder, "SAIL_annotation_screenplays/line_indices.txt")).read().splitlines()]
    
    #####################################################################
    #### set test movie
    #### if test movie is all do leave one movie out cross val
    #### exit if test movie is not found in movie list
    #####################################################################
    
    if eval_movie in movies:
        movies = [eval_movie]
    elif eval_movie == "all":
        pass
    else:
        print("movie not found")
        sys.exit(-1)

    #####################################################################
    #### if training in parallel, set number of processes to spawn
    #####################################################################
    
    if parallel:
        n_gpus = torch.cuda.device_count()
        n_processes = n_gpus * n_folds_per_gpu
    else:
        n_gpus = 1
        n_processes = 1
    
    #####################################################################
    #### find number of iterations of parallel execution
    #### find set of common arguments
    #####################################################################
    
    n_iterations = math.ceil(len(movies)/n_processes)
    arguments = [results_folder, seqlen, bidirectional, train_batch_size, segment_lengths, encoder_learning_rate, learning_rate, max_epochs, max_norm, verbose]

    #####################################################################
    #### save the output dictionary from each fold in a list
    #####################################################################
    
    output_dicts = []

    #####################################################################
    #### set subprocess create method to spawn
    #####################################################################
    
    print("starting multiprocess training:\n")
    set_start_method('spawn', force=True)

    #####################################################################
    #### iterate over batch of parallel subprocesses
    #####################################################################
    
    for i in range(n_iterations):
        process_movies = movies[i * n_processes: (i + 1) * n_processes]
        process_devices = [torch.device(j % n_gpus) for j in range(len(process_movies))]
        process_arguments = [tuple([process_movies[j], process_devices[j]] + arguments) for j in range(len(process_movies))]
        
        #####################################################################
        #### start process Pool
        #####################################################################
        
        with Pool() as pool:
            process_output_dicts = pool.starmap(train_movie, process_arguments)
        
        #####################################################################
        #### collect output dictionary from each process in Pool
        #####################################################################
        
        output_dicts.extend(process_output_dicts)
    
    print("\n\nmultiprocess training ends")
    
    #####################################################################
    #### combine output dicts by epoch and segment length
    #####################################################################
    
    combined_output_dict = {}

    for epoch in range(max_epochs):

        combined_output_dict[f"epoch_{epoch + 1}"] = {}

        for segment_length in segment_lengths:

            #####################################################################
            #### collect label and pred array from each fold
            #####################################################################
            
            label, pred = [], []

            for output_dict in output_dicts:
                label.extend(output_dict[f"epoch_{epoch + 1}"][segment_length]["label"])
                pred.extend(output_dict[f"epoch_{epoch + 1}"][segment_length]["pred"])

            #####################################################################
            #### save combined label and pred list
            #### and calculate per class performance and micro scores
            #####################################################################
            
            combined_output_dict[f"epoch_{epoch + 1}"][segment_length] = {"label": label, "pred": pred}
            combined_output_dict[f"epoch_{epoch + 1}"][segment_length]["per_class_performance"] = get_classification_report(label, pred).to_dict()
            p, r, f, _ = precision_recall_fscore_support(label, pred, zero_division=0, labels=list("SNCDET"), average="micro")
            combined_output_dict[f"epoch_{epoch + 1}"][segment_length]["micro"] = {"precision": p, "recall": r, "f1": f}

    #####################################################################
    #### save to results/cross_val/seqlen<int>_lomo/<movie-name>_bi<True/False>.json
    #####################################################################
    
    filepath = os.path.join(results_folder, "cross_val", f"seqlen{seqlen}_lomo-{eval_movie}_bi{bidirectional}.json")
    json.dump(combined_output_dict, open(filepath, "w"), indent=2)