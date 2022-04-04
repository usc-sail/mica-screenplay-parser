# author : Sabyasachee

# standard library imports
from typing import Tuple, List

# third party imports
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, f1_score

# user library imports
from movieparser.scriptparser import ScriptParser
from movieparser.scriptloader import ScriptLoader, label2id

def get_classification_report(label, pred) -> pd.DataFrame:
    C = confusion_matrix(label, pred, labels=list(range(len(label2id))))
    precision, recall, f1, support = precision_recall_fscore_support(label, pred, zero_division=0, labels=list(range(len(label2id))))
    macro_f1 = f1_score(label, pred, zero_division=0, average="macro", labels=list(range(1, len(label2id))))
    micro_f1 = f1_score(label, pred, zero_division=0, average="micro", labels=list(range(1, len(label2id))))
    wtavg_f1 = f1_score(label, pred, zero_division=0, average="weighted", labels=list(range(1, len(label2id))))
    id2label = dict((i, label) for label, i in label2id.items())
    labels = [id2label[i] for i in range(C.shape[0])]
    df = pd.DataFrame(C, columns=labels, index=labels)
    df["support"] = support
    df["precision"] = precision
    df["recall"] = recall
    df["f1"] = f1
    df["macro-f1"] = macro_f1
    df["micro-f1"] = micro_f1
    df["weighted-f1"] = wtavg_f1
    return df

def evaluate(parser: ScriptParser, loader: ScriptLoader) -> Tuple[pd.DataFrame, float]:
    parser.eval()
    label, pred, losses = [], [], []
    
    with torch.no_grad():
        for eval_scripts, eval_features, eval_labels in loader:
            loss, _pred = parser(eval_scripts, eval_features, eval_labels)
            label.append(eval_labels)
            pred.append(_pred)
            losses.append(loss.cpu().detach().item())
    
    label = torch.cat(label).cpu().numpy().astype(int).flatten()
    pred = torch.cat(pred).cpu().numpy().astype(int).flatten()
    avg_loss = np.mean(losses)

    return get_classification_report(label, pred), avg_loss

def evaluate_movie(parser: ScriptParser, loader: List[Tuple[List[str], List[List[float]], List[int]]]) -> pd.DataFrame:
    parser.eval()
    device = next(parser.parameters()).device
    label, pred = [], []
    with torch.no_grad():
        for eval_scripts, eval_features, eval_labels in tqdm(loader):
            features = torch.FloatTensor(eval_features).to(device)
            _pred = parser.parse(eval_scripts, features)
            label.extend(eval_labels)
            pred.extend(_pred)
    return get_classification_report(label, pred)