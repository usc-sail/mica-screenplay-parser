# author : Sabyasachee

# standard library imports
from collections import defaultdict
import math
import os
from typing import Tuple

# third party imports
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

label2id = defaultdict(int)
for i, label in enumerate("SNCDTEM"):
    label2id[label] = i + 1

class ScriptLoader:

    def __init__(self, scripts: np.ndarray, features: torch.FloatTensor, labels: torch.LongTensor, \
        batch_size: int, shuffle: bool=False) -> None:
        self.scripts = scripts
        self.features = features
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self) -> int:
        return math.ceil(len(self.scripts)/self.batch_size)
    
    def __iter__(self):
        if self.shuffle:
            index = np.random.permutation(len(self.scripts))
            self.scripts = self.scripts[index]
            self.features = self.features[index]
            self.labels = self.labels[index]
        self.i = 0
        return self
    
    def __next__(self) -> Tuple[np.ndarray, torch.FloatTensor, torch.LongTensor]:
        if self.i < len(self):
            batch_index = slice(self.i * self.batch_size, (self.i + 1) * self.batch_size)
            batch_scripts = self.scripts[batch_index]
            batch_features = self.features[batch_index]
            batch_labels = self.labels[batch_index]
            self.i += 1
            return batch_scripts, batch_features, batch_labels
        else:
            raise StopIteration
    
def get_dataloaders(results_folder: str, seqlen: int, train_batch_size: int, eval_batch_size: int, \
    eval_movie: str = None, leave_one_movie_out: bool = False, device: torch.device = "cpu") -> \
        Tuple[ScriptLoader, ScriptLoader, ScriptLoader]:

    df = pd.read_csv(os.path.join(results_folder, "seq_{}.csv".format(seqlen)), index_col=None)
    data_df = pd.read_csv(os.path.join(results_folder, "data.csv"), index_col=None)
    feats_df = pd.read_csv(os.path.join(results_folder, "feats.csv"), index_col=0)
    features_file = os.path.join(results_folder, "seq_{}_feats.pt".format(seqlen))
    scripts, features, labels = [], [], []

    if os.path.exists(features_file):
        features = torch.load(features_file).float()
    else:
        features = []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            feature = [feats_df.loc[row["line_{}".format(i + 1)]] for i in range(seqlen)]
            features.append(feature)
        features = torch.FloatTensor(features)
        torch.save(features, features_file)
        
    scripts = df[["line_{}".format(i + 1) for i in range(seqlen)]].values
    labels = df["label"].apply(lambda labelseq: [label2id[label] for label in labelseq])
    
    scripts = np.array(scripts)
    features = features.to(device)
    labels = torch.LongTensor(labels).to(device)

    if eval_movie is None:
        train_index = (df["split"] == "train").values
        test_index = (df["split"] == "test").values
        dev_index = (df["split"] == "dev").values
        test_loader = ScriptLoader(scripts[test_index], features[test_index], labels[test_index], eval_batch_size)
        dev_loader = ScriptLoader(scripts[dev_index], features[dev_index], labels[dev_index], eval_batch_size)
    else:
        train_index = (df["movie"] != eval_movie).values
        if leave_one_movie_out:
            eval_scripts, eval_features, eval_labels = [], [], []
            for _, edf in data_df[data_df["movie"] == eval_movie].groupby("error"):
                eval_scripts.append(edf["text"].tolist())
                eval_features.append([feats_df.loc[text].tolist() for text in edf["text"]])
                eval_labels.append(edf["label"].apply(lambda label: label2id[label]).tolist())
            test_loader = dev_loader = list(zip(eval_scripts, eval_features, eval_labels))
        else:
            test_index = dev_index = (df["movie"] == eval_movie).values
            test_loader = dev_loader = ScriptLoader(scripts[test_index], features[test_index], \
                labels[test_index], eval_batch_size)
    
    train_loader = ScriptLoader(scripts[train_index], features[train_index], labels[train_index], train_batch_size, \
        shuffle=True)

    return train_loader, test_loader, dev_loader