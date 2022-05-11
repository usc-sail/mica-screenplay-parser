# author : Sabyasachee

'''
Screenplay Parser command-line interface reads screenplay text/pdf files and outputs csv files
containing screenplay lines and tags. Tags can be:

    1. S - Scene Header
    2. N - Scene Description
    3. C - Character
    4. D - Utterance
    5. T - Transition
    6. E - Expression
    7. M - Metadata
    8. O - Other

Usage:
    screenplayparser <input> [<output> --rules --keep_pdf2text --gpu=<gpu_id>]

    <input> is path to either a directory, text file, or pdf file
    <output> is path to either a directory, csv file, or is left unspecified

    If <input> is path to a text file, the program reads the text file, parses it, and saves 
    the parsed output to a csv file in path <output>
        If <output> is unspecified, program saves the parsed output to a csv file in the same 
        directory as <input>. The filename is same as <input> but "_parsed" is appended
    
    If <input> is path to a pdf file, the program first converts the pdf to a text file, and
    proceeds as above. If --keep_pdf2text is set, the converted text file is also saved. The 
    converted text file is saved to the same directory as <input>. The filename is same as 
    <input>.

    If <input> is a directory, the program first scans it for text and pdf files, and proceeds
    as above. <output> has to be an existing directory. The csv files containing the parsed 
    output and the converted text files (if pdf files exist and --keep_pdf2text is set) are saved
    to <output> directory.

Options:
    --rules                 Use rules to parse screenplay. parsing is faster but less accurate.
                            By default, the ML parser is used, which is slower but more accurate.
    --keep_pdf2text         If input_file is a pdf, don't remove the text file converted from the pdf
    --gpu=<gpu_id>          Use gpu. If --rules is set, then gpu is never used [default: -1]
'''

# standard library
import os
import re
import sys

# third party
import docopt
import pandas as pd
import pdftotext
from tqdm import tqdm

# user
from screenplayparser import ScreenplayParser

#####################################################################
#### read cmd args
#####################################################################

args = docopt.docopt(__doc__)

input = args["<input>"]
output = args["<output>"]

#####################################################################
#### read input_files
#####################################################################

input_files = []
output_files = []

if os.path.isdir(input):
    input_files = [file for file in os.listdir(input) if file.endswith(".txt") or file.endswith(".pdf")]
    
    if os.path.isdir(output):
        output_files = [re.sub("\.(txt|pdf)$", "_parsed.csv", file) for file in input_files]
    else:
        print("<output> has to be a directory")
        sys.exit(-1)

elif input.endswith(".txt") or input.endswith(".pdf"):
    input_files = [input]
    
    if output is None:
        output = re.sub("\.(txt|pdf)$", "_parsed.csv", input)
    output_files = [output]

else:
    print("<input> must be a path to a directory or a text/pdf file")
    sys.exit(-1)

#####################################################################
#### initialize parser
#####################################################################

if args["--rules"]:
    parser = ScreenplayParser(use_rules=True)
else:
    parser = ScreenplayParser(device_id=int(args["--gpu"]))

#####################################################################
#### parse files
#####################################################################

for input_file, output_file in tqdm(input_files):

    script = []

    if input_file.endswith(".pdf"):
        with open(input_file, "rb") as fr:
            pdf = pdftotext.PDF(fr)
            text = "\n\n".join(pdf)

            if args["--keep_pdf2txt"]:
                converted_text_file = re.sub("\.pdf$", ".txt", input_file)
                with open(converted_text_file, "w") as fw:
                    fw.write(converted_text_file)
            
            script = text.splitlines()
    
    else:
        with open(input_file, "r") as fr:
            script = fr.read().splitlines()
    
    tags = parser.parse(script)

    df = pd.DataFrame()
    df["text"] = script
    df["label"] = tags
    df.to_csv(output_file, index=False, sep=",")