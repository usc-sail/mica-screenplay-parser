# author : Sabyasachee

# standard library imports
from typing import List, Dict, Any

# third party imports
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

# user library imports
from movieparser.scriptparser import ScriptParser
from movieparser.scriptloader import InferenceScriptLoader, label2id

def get_classification_report(label: List[str], pred: List[str]) -> pd.DataFrame:
    C = confusion_matrix(label, pred, labels=list(label2id.keys()))
    precision, recall, f1, support = precision_recall_fscore_support(label, pred, zero_division=0, labels=list(label2id.keys()))
    id2label = dict((i, label) for label, i in label2id.items())
    labels = [id2label[i] for i in range(C.shape[0])]
    df = pd.DataFrame(C, columns=labels, index=labels)
    df["support"] = support
    df["precision"] = precision
    df["recall"] = recall
    df["f1"] = f1
    return df

def evaluate(parser: ScriptParser, loader: InferenceScriptLoader, process_name: str = "", segment_lengths: List[int] = [10, 50, 100, 1000000]) -> Dict[str, Any]:
    '''
    return a dictionary of results indexed by segment lengths \\
    if segment length is None, key in dictionary is 1000000 (very large number)

    the returned dictionary has the following format: \\
        segment_length_1:
            label: List[str]
            pred: List[str]
            per_class_performance: pd.DataFrame \\
            micro:
                precision: float
                recall: float
                f1: float
        segment_length_2:
            ...
    
    the per_class_performance dataframe is 8 x 12 matrix
    the row index is O, S, N, C, D, E, T, M
    the column index is the row index + support, precision, recall, f1

    the micro scores are only for S, N, C, D, E, T
    we ignore O and M
    '''
    parser.eval()
    eval_dict = {}

    with torch.no_grad():
        for segment_length in segment_lengths:
            
            print("{:25s}: evaluating with segment_length = {}".format(process_name, segment_length))

            if segment_length is None:
                key = 1000000
            else:
                key = segment_length
            eval_dict[key] = {"label": [], "pred": [], "per_class_performance": None, "micro": None}
            
            for eval_scripts, eval_labels in loader:
                pred = parser.parse(eval_scripts, segment_length=segment_length)
                eval_dict[key]["label"].extend(eval_labels)
                eval_dict[key]["pred"].extend(pred)
    
            eval_dict[key]["per_class_performance"] = get_classification_report(eval_dict[key]["label"], eval_dict[key]["pred"])
            p, r, f, _ = precision_recall_fscore_support(eval_dict[key]["label"], eval_dict[key]["pred"], zero_division=0, labels=list("SNCDET"), average="micro")
            eval_dict[key]["micro"] = {"precision": p, "recall": r, "f1": f}
    
    return eval_dict