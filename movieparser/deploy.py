# author : Sabyasachee

# standard library imports
import os

# third party imports
import numpy as np
import torch
from torch.optim import Adam

# user library imports
from movieparser.scriptparser import ScriptParser
from movieparser.scriptloader import get_dataloaders, label2id

def deploy(results_folder: str, seqlen: int, bidirectional: bool, train_batch_size: int, learning_rate: float, encoder_learning_rate: float, max_norm: float, max_epochs: int):
    device = torch.device(0) 
    train_loader, _, _ = get_dataloaders(results_folder, seqlen, train_batch_size, 1, "<DUMMY MOVIE>", device)
    feature_size = train_loader.features.shape[2]
    scriptparser = ScriptParser(feature_size, len(label2id), bidirectional)
    scriptparser.to(device)

    optimizer = Adam([
        {"params": scriptparser.encoder.parameters(), "lr": encoder_learning_rate},
        {"params": [parameter for name, parameter in scriptparser.named_parameters() if not name.startswith("encoder")]}
    ], lr=learning_rate)

    for epoch in range(max_epochs):
        print("epoch {:2d}".format(epoch + 1))
        train_losses = []
        scriptparser.train()

        for b, (train_scripts, train_features, train_labels) in enumerate(train_loader):
            loss, _ = scriptparser(train_scripts, train_features, train_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(scriptparser.parameters(), max_norm=max_norm)
            optimizer.step()
            optimizer.zero_grad()
            train_losses.append(loss.detach().cpu().item())
            if (b + 1) % 20 == 0:
                print("epoch {:2d} batch {:3d} avg. train loss = {:.3f}".format(epoch + 1, b + 1, np.mean(train_losses)))
        
        torch.save(scriptparser.state_dict(), os.path.join(results_folder, "saved_models", "epoch{}.pt".format(epoch + 1)))