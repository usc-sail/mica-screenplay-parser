from collections import defaultdict
import math
import os
from typing import Tuple, List
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

label2id = defaultdict(int)
for i, label in enumerate("OSNCDETM"):
    label2id[label] = i

class ScriptLoader:

    def __init__(self, scripts: np.ndarray, features: torch.FloatTensor, labels: torch.LongTensor, batch_size: int, shuffle: bool=False) -> None:
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
        raise StopIteration
    
class InferenceScriptLoader:

    def __init__(self, df: pd.DataFrame, movie: str) -> None:
        scripts, labels = [], []
        for _, edf in df[df["movie"] == movie].groupby("error"):
            scripts.append(edf["text"].fillna("").tolist())
            labels.append(edf["label"].tolist())
        self.scripts = scripts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.scripts)
    
    def __iter__(self):
        self.i = 0
        return self
    
    def __next__(self) -> Tuple[List[str], List[str]]:
        if self.i < len(self):
            self.i += 1
            return self.scripts[self.i - 1], self.labels[self.i - 1]
        raise StopIteration

def get_dataloaders(results_folder: str, seqlen: int, train_batch_size: int, eval_movie: str, device: torch.device = "cpu") -> Tuple[ScriptLoader, InferenceScriptLoader]:

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
    

    scripts = df[["line_{}".format(i + 1) for i in range(seqlen)]].fillna("").values
    labels = df["label"].apply(lambda labelseq: [label2id[label] for label in labelseq])
    
    scripts = np.array(scripts).astype(str)
    features = features.to(device)
    labels = torch.LongTensor(labels).to(device)

    if eval_movie not in df["movie"].tolist():
        print("eval movie = {} not in seq_{}.csv, training on all movies".format(eval_movie, seqlen))
    
    train_index = (df["movie"] != eval_movie).values
    train_loader = ScriptLoader(scripts[train_index], features[train_index], labels[train_index], train_batch_size, shuffle=True)
    test_loader = InferenceScriptLoader(data_df, eval_movie)

    return train_loader, test_loader