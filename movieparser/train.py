# author : Sabyasachee

# standard library imports

# third party imports
import numpy as np
import pandas as pd
import torch
from torch.optim import Adam

# user imports
from movieparser.scriptloader import get_dataloaders, label2id
from movieparser.scriptparser import ScriptParser
from movieparser.evaluate import evaluate, evaluate_movie

def train(results_folder: str, seqlen: int, train_batch_size: int, eval_batch_size: int, eval_movie: str, \
    leave_one_movie_out: bool, learning_rate: float, encoder_learning_rate: float, max_epochs: int, patience: int, max_norm: float):
    device = torch.device("cuda:0")
    train_loader, test_loader, dev_loader = get_dataloaders(results_folder, seqlen, train_batch_size, \
        eval_batch_size, eval_movie, leave_one_movie_out, device)
    print("{} train batches, {} dev batches, {} test batches".format(len(train_loader), len(dev_loader), \
        len(test_loader)))

    feature_size = train_loader.features.shape[2]
    scriptparser = ScriptParser(feature_size, len(label2id))
    scriptparser.to(device)

    optimizer = Adam([
        {"params": scriptparser.encoder.parameters(), "lr": encoder_learning_rate},
        {"params": [parameter for name, parameter in scriptparser.named_parameters() if not name.startswith("encoder")]}
    ], lr=learning_rate)

    for epoch in range(max_epochs):
        print("epoch {}".format(epoch + 1))
        train_losses = []
        scriptparser.train()
        for b, (train_scripts, train_features, train_labels) in enumerate(train_loader):
            loss, _ = scriptparser(train_scripts, train_features, train_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(scriptparser.parameters(), max_norm=max_norm)
            optimizer.step()
            optimizer.zero_grad()
            train_losses.append(loss.detach().cpu().item())
            if (b + 1) % 50 == 0:
                print("\t batch {:3d} avg. train loss = {:.3f}".format(b + 1, np.mean(train_losses)))
        
        print("dev evaluation:")
        if eval_movie is None or not leave_one_movie_out:
            dev_perf = evaluate(scriptparser, dev_loader)
        else:
            dev_perf = evaluate_movie(scriptparser, dev_loader)
        print(dev_perf)
        print()
    
    print()
    print("test evaluation:")
    if eval_movie is None or not leave_one_movie_out:
        test_perf = evaluate(scriptparser, test_loader)
    else:
        test_perf = evaluate(scriptparser, test_loader)
    print(test_perf)