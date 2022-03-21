# author : Sabyasachee

"""
Robust Movie Screenplay Parser

Usage:
    movieparser evaluate        [   --data=<DATA> ]
    movieparser evaluate robust [   --data=<DATA> --results=<RESULTS> --names_file=<path> ]
    movieparser evaluate gdi    [   --data=<DATA> --gdi_folders=<FOLDERS> ]
    movieparser create_data     [   --data=<DATA> --results=<RESULTS> --names_file=<path> ]
    movieparser create_seq_data [   --results=<RESULTS> --seqlen=<int>]

Options:
    -h, --help                                      Show this help screen and exit
        --data=<DATA>                               path to data folder [default: data]
        --results=<RESULTS>                         path to results folder [default: results]
        --gdi_folders=<FOLDERS>                     comma-separated list of GDI folders 
                                                    [default: LEGO TITAN,Lionsgate, NBC Universal]
        --names_file=<path>                         file containing English names [default: data/names.txt]
        --seqlen=<int>                              number of sentences in a sample [default: 10]
"""

# standard library
from docopt import docopt

def read_args():
    cmd_args = docopt(__doc__)
    args = {}

    args["data"] = cmd_args["--data"]
    args["results"] = cmd_args["--results"]
    args["gdi_folders"] = cmd_args["--gdi_folders"].split(",")
    args["names_file"] = cmd_args["--names_file"]
    args["seqlen"] = int(cmd_args["--seqlen"])

    if cmd_args["evaluate"]:
        if cmd_args["gdi"]:
            args["mode"] = "evaluate-gdi"
        elif cmd_args["robust"]:
            args["mode"] = "evaluate-parser-robust"
        else:
            args["mode"] = "evaluate-parser"
    
    elif cmd_args["create_data"]:
        args["mode"] = "create_data"

    elif cmd_args["create_seq_data"]:
        args["mode"] = "create_seq_data"

    return args