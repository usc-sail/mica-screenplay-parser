# author : Sabyasachee

"""
Robust Movie Screenplay Parser

Usage:
    movieparser evaluate        [   --data=<DATA> ]
    movieparser evaluate robust [   --data=<DATA> --results=<RESULTS> --names_file=<path> ]
    movieparser evaluate gdi    [   --data=<DATA> --gdi_folders=<FOLDERS> ]

Options:
    -h, --help                                      Show this help screen and exit
        --data=<DATA>                               path to data folder [default: data]
        --results=<RESULTS>                         path to results folder [default: results]
        --gdi_folders=<FOLDERS>                     comma-separated list of GDI folders 
                                                    [default: LEGO TITAN,Lionsgate, NBC Universal]
        --names_file=<path>                         file containing English names [default: data/names.txt]
"""

# standard library
from docopt import docopt

def read_args():
    cmd_args = docopt(__doc__)
    args = {}

    args["data"] = cmd_args["--data"]
    args["results"] = cmd_args["--results"]

    if cmd_args["evaluate"]:
        
        if cmd_args["gdi"]:
            args["mode"] = "evaluate-gdi"
            args["gdi_folders"] = cmd_args["--gdi_folders"].split(",")
        
        elif cmd_args["robust"]:
            args["mode"] = "evaluate-parser-robust"
            args["names_file"] = cmd_args["--names_file"]

        else:
            args["mode"] = "evaluate-parser"

    return args