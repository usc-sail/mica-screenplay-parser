# author : Sabyasachee

"""
Robust Movie Screenplay Parser

Usage:
    movieparser evaluate            [--data=<DATA>]
    movieparser evaluate robust     [--data=<DATA>] [--results=<RESULTS>] [--names_file=<path>] [--all]
    movieparser evaluate gdi        [--data=<DATA>] [--gdi_folders=<FOLDERS>] [--ignore_scripts=<SCRIPTS>]
    movieparser create data         [--data=<DATA>] [--results=<RESULTS>] [--names_file=<path>]
    movieparser create seq          [--results=<RESULTS>] [--seqlen=<int>]
    movieparser create feats        [--results=<RESULTS>]
    movieparser train               [--results=<RESULTS>] [--seqlen=<int>] [--train_batch_size=<int>] 
                                    [--eval_movie=<str>] [--eval_segment_lengths=<int_array>] [--learning_rate=<float>] 
                                    [--enc_learning_rate=<float>] [--max_norm=<float>]
                                    [--max_epochs=<int>] [--parallel] [--n_folds_per_gpu=<int>] [--bi] [--verbose]
    movieparser deploy              [--results=<RESULTS>] [--seqlen=<int>] [--train_batch_size=<int>] [--learning_rate=<float>]
                                    [--enc_learning_rate=<float>] [--max_norm=<float>] 
                                    [--max_epochs=<int>] [--bi]

Options:
    -h, --help                                      Show this help screen and exit
        --data=<DATA>                               path to data folder [default: data]
        --results=<RESULTS>                         path to results folder [default: results]
        --all                                       set to evaluate on different error probabilities
        --gdi_folders=<FOLDERS>                     comma-separated list of GDI folders 
                                                    [default: LEGO TITAN,Lionsgate,NBC Universal]
        --ignore_scripts=<SCRIPTS>                  comma-separated scripts to ignore because of annotation error
                                                    script name is GDI_folder_name/script_name e.g. LEGO TITAN/EPISODE 102
                                                    [default: ]
        --names_file=<path>                         file containing English names [default: data/names.txt]
        --seqlen=<int>                              number of sentences in a sample [default: 10]
        --train_batch_size=<int>                    training batch size [default: 256]
        --eval_movie=<str>                          movie left out in leave-one-movie-out testing [default: all]
        --eval_segment_lengths=<int_array>          comma-separated list of integer segment lengths, 
                                                    the very large number at the end signifies that the entire script 
                                                    will be parsed at once instead of in segments
                                                    [default: 10,50,100,1000000]
    -l, --learning_rate=<float>                     learning rate [default: 1e-3]
        --enc_learning_rate=<float>                 learning rate of the sentence encoder [default: 1e-5]
        --max_norm=<float>                          maximum weight norm for clipping [default: 1.0]
        --max_epochs=<int>                          maximum number of epochs [default: 5]
        --parallel                                  start training on multiple folds
        --n_folds_per_gpu=<int>                     number of simultaneous folds to train in a single gpu [default: 1]
        --bi                                        bidirectional
    -v, --verbose                                   verbose logging during training
"""

# standard library
from docopt import docopt

def read_args():
    cmd_args = docopt(__doc__)
    args = {}

    args["data"] = cmd_args["--data"]
    args["results"] = cmd_args["--results"]
    args["gdi_folders"] = cmd_args["--gdi_folders"].split(",")
    args["ignore_scripts"] = cmd_args["--ignore_scripts"].split(",") if cmd_args["--ignore_scripts"].strip() != "" else []
    args["names_file"] = cmd_args["--names_file"]
    args["seqlen"] = int(cmd_args["--seqlen"])
    args["train_batch_size"] = int(cmd_args["--train_batch_size"])
    args["eval_movie"] = None if cmd_args["--eval_movie"] == "" else cmd_args["--eval_movie"]
    args["eval_segment_lengths"] = [int(l) for l in cmd_args["--eval_segment_lengths"].split(",")]
    args["learning_rate"] = float(cmd_args["--learning_rate"])
    args["enc_learning_rate"] = float(cmd_args["--enc_learning_rate"])
    args["max_norm"] = float(cmd_args["--max_norm"])
    args["max_epochs"] = int(cmd_args["--max_epochs"])
    args["parallel"] = cmd_args["--parallel"]
    args["n_folds_per_gpu"] = int(cmd_args["--n_folds_per_gpu"])
    args["bidirectional"] = cmd_args["--bi"]
    args["verbose"] = cmd_args["--verbose"]

    if cmd_args["evaluate"]:
        if cmd_args["gdi"]:
            args["mode"] = "evaluate-gdi"
        elif cmd_args["robust"]:
            args["mode"] = "evaluate-parser-robust"
        else:
            args["mode"] = "evaluate-parser"
    
    elif cmd_args["create"]:
        if cmd_args["data"]:
            args["mode"] = "create-data"

        elif cmd_args["seq"]:
            args["mode"] = "create-seq"

        elif cmd_args["feats"]:
            args["mode"] = "create-feats"
    
    elif cmd_args["train"]:
        args["mode"] = "train"
    
    elif cmd_args["deploy"]:
        args["mode"] = "deploy"

    return args