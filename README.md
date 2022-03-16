# Robust Screenplay Parser

## Requirements

Create a conda environment and install requirements using:

```
conda create -n parser python=3.8
pip install -r requirements.txt
```

## Usage

```
Robust Movie Screenplay Parser

Usage:
    movieparser evaluate        [   --data=<DATA> ]
    movieparser evaluate robust [   --data=<DATA> --names_file=<path> ]
    movieparser evaluate gdi    [   --data=<DATA> --gdi_folders=<FOLDERS> ]

Options:
    -h, --help                                      Show this help screen and exit
        --data=<DATA>                               path to data folder [default: data]
        --gdi_folders=<FOLDERS>                     comma-separated list of GDI folders 
                                                    [default: LEGO TITAN,Lionsgate, NBC Universal]
        --names_file=<path>                         file containing English names [default: data/names.txt]
```