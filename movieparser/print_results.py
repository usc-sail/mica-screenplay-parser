from movie_screenplay_parser.movieparser.parse_scripts_noindent import parse_lines

import json
import os
import sys
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

def print_results(results_folder):
    df = pd.read_csv(os.path.join(results_folder, "data.csv"), index_col=None)
    result = json.load(open(os.path.join(results_folder, "cross_val/seqlen100_lomo-all_biTrue.json")))

    true = []
    pred = []

    for _, medf in df.groupby(["error", "movie"]):
        try:
            tags = parse_lines(medf["text"].fillna(""))
            pred.extend(tags)
            true.extend(medf["label"].tolist())
        except Exception as e:
            pass

    print("rule:")
    p, r, f1, _ = precision_recall_fscore_support(true, pred, labels=list("SNCDET"), zero_division=0, average="micro")
    print("micro: prec = {:.3f}, rec = {:.3f}, f1 = {:.3f}".format(p, r, f1))
    p, r, f1, _ = precision_recall_fscore_support(true, pred, labels=list("SNCDET"), zero_division=0)
    df = pd.DataFrame()
    df["tag"] = list("SNCDET")
    df["precision"] = p
    df["recall"] = r
    df["f1"] = f1
    df.index = df["tag"]
    print(df)
    print()

    print("robust:")
    micro = result["epoch_6"]["1000000"]["micro"]
    p, r, f1 = micro["precision"], micro["recall"], micro["f1"]
    print("micro: prec = {:.3f}, rec = {:.3f}, f1 = {:.3f}".format(p, r, f1))
    df = pd.DataFrame.from_dict(result["epoch_6"]["1000000"]["per_class_performance"])
    df = df.loc[list("SNCDET"), ["precision", "recall", "f1"]]
    print(df)