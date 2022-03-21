# author : Sabyasachee

# standard library
import math
import os
import random

# third party library
import pandas as pd

def create_seq_data(results_folder, seqlen=10):

    random.seed(0)
    
    data_file = os.path.join(results_folder, "data.csv")
    df = pd.read_csv(data_file, index_col=None)
    df = df.sort_values(by=["movie", "error", "line_no"])

    data = []
    header = ["movie", "start_line_no", "end_line_no"] + ["line_{}".format(i + 1) for i in range(seqlen)] + ["label", "error"]

    for (movie, error), mdf in df.groupby(["movie", "error"]):
        lines = mdf["text"].values
        label = mdf["label"].values
        for i in range(len(lines) - seqlen + 1):
            sample_lines = lines[i: i + seqlen].tolist()
            sample_label = label[i: i + seqlen].tolist()
            data.append([movie, i + 1, i + seqlen] + sample_lines + ["".join(sample_label), error])
    
    df = pd.DataFrame(data, columns=header)
    df["priority"] = df["error"] != "NONE"
    df = df.sort_values(by=["movie", "priority", "error", "start_line_no"])
    df = df.drop_duplicates(subset=["movie"] + ["line_{}".format(i + 1) for i in range(seqlen)], keep="first")
    df = df.drop(columns=["priority"])

    df["split"] = "train"
    for _, mdf in df.groupby(["movie", "error"]):
        n = len(mdf)
        if n == 1:
            arr = ["test"]
        else:
            n_test = math.ceil(0.1 * n)
            n_train = n - 2 * n_test
            arr = n_test * ["test"] + n_test * ["dev"] + n_train * ["train"]
            random.shuffle(arr)
        df.loc[mdf.index, "split"] = arr
    
    n_train = (df["split"] == "train").sum()
    n_test = (df["split"] == "test").sum()
    n_dev = (df["split"] == "dev").sum()
    print("{} train, {} dev, {} test".format(n_train, n_dev, n_test))

    seq_file = os.path.join(results_folder, "seq_{}.csv".format(seqlen))
    df.to_csv(seq_file, index=False)