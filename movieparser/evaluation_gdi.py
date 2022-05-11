# author : Sabyasachee

# standard library
from collections import Counter
import os
import re

# third party
from docx import Document
import numpy as np
import pandas as pd
from tqdm import tqdm

# user library imports
from movieparser.parse_scripts_noindent import parse_lines
from movieparser.robust_parser import MovieParser

def evaluate_gdi(gdi_folder, gdi_folder_names, ignore_scripts=[], use_robust_parser=False, ignore_existing_parse=False, recalculate_line_counts=False):

    #####################################################################
    #### get GDI annotated character line counts
    #####################################################################

    gdi_count_filepath = os.path.join(gdi_folder, "gdi_line_counts.tsv")

    if os.path.exists(gdi_count_filepath) and not recalculate_line_counts:
        line_count_df = pd.read_csv(gdi_count_filepath, index_col=None, sep="\t")
    else:
        data = []

        for folder in gdi_folder_names:
            movies = sorted([re.sub("\.txt$", "", f) for f in os.listdir(os.path.join(gdi_folder, "{}/Scripts_Txt/".format(folder))) if f.endswith(".txt")])

            for movie in movies:
                analysis_docx = os.path.join(os.path.join(gdi_folder, "{}/Analysis/{}.docx".format(folder, movie)))

                if "{}/{}".format(folder, movie) not in ignore_scripts and os.path.exists(analysis_docx):
                    
                    with open(analysis_docx, "rb") as f:
                        doc = Document(f)
                        table = doc.tables[0]

                        i = 2
                        cells = [cell.text for row in table.rows for cell in row.cells]

                        while i + 1 < len(cells):
                            data.append([folder, movie, cells[i].strip().upper(), cells[i + 1]])
                            i += 2

        line_count_df = pd.DataFrame(data, columns=["folder", "movie", "character", "line-count"])
        line_count_df.to_csv(os.path.join(gdi_folder, "gdi_line_counts.tsv"), sep="\t", index=False)

    #####################################################################
    #### get the unique gdi_folder_name/movie items
    #####################################################################
    
    items = []

    for _, row in line_count_df[["folder", "movie"]].drop_duplicates().iterrows():
        items.append((row["folder"], row["movie"]))
    
    print("{} items".format(len(items)))

    #####################################################################
    #### get SAIL script parser character line counts
    #### remove leading/trailing space, digits, and '*'
    #### remove trailing paranthesized expressions
    #### character name should contain atleast 3 letters
    #### character name should be followed by O, D, or E
    #####################################################################
    
    if use_robust_parser:
        line_count_filepath = os.path.join(gdi_folder, "robust_parser_line_counts.tsv")
    else:
        line_count_filepath = os.path.join(gdi_folder, "parser_line_counts.tsv")

    if os.path.exists(line_count_filepath) and not ignore_existing_parse and not recalculate_line_counts:
        sys_line_count_df = pd.read_csv(line_count_filepath, index_col=None, sep="\t")

    else:
        data = []

        parser = None

        for folder, movie in tqdm(items):
            slines = open(os.path.join(gdi_folder, "{}/Scripts_Txt/{}.txt".format(folder, movie))).read().splitlines()
            
            if use_robust_parser:
                parsed_filepath = os.path.join(gdi_folder, "{}/ParsedRobust/{}_tags.txt".format(folder, movie))
            else:
                parsed_filepath = os.path.join(gdi_folder, "{}/Parsed/{}_tags.txt".format(folder, movie))

            if os.path.exists(parsed_filepath) and not ignore_existing_parse:
                tags = open(parsed_filepath).read().splitlines()
                print(f"{folder:20s}/{movie:20s}: {len(slines)} {len(tags)}")
            else:
                if use_robust_parser:
                    if parser is None:
                        parser = MovieParser()
                    tags = parser.parse(slines)
                else:
                    tags = parse_lines(slines)
                open(parsed_filepath, "w").write("\n".join(tags))
            tags = [t if t in ["S","N","C","D","E","T","M"] else "O" for t in tags]

            characters = []
            n = len(slines)
            i = 0

            while i < n:
                if tags[i] == "C":
                    character = re.sub("(^[\s\d\*]*)|((\(.+\)\s*)*[\s\d\*]*$)", "", slines[i]).strip().upper()

                    if len(character) > 2:
                        tcharacters = re.split("\s{2,}|/", character)
                        i += 1
                        while i < n and tags[i] not in ["C", "S"]:
                            if tags[i] == "D":
                                characters.extend(tcharacters)
                            i += 1
                        i -= 1
                i += 1

            citems = sorted(Counter(characters).items(), key = lambda x: x[1], reverse=True)
            for ch, cn in citems:
                data.append([folder, movie, ch, cn])

        sys_line_count_df = pd.DataFrame(data, columns=["folder", "movie", "character", "line-count"])
        sys_line_count_df.to_csv(line_count_filepath, sep="\t", index=False)

    #####################################################################
    #### evaluate character identification and line count errors
    #### on correctly identified characters of script parser against
    #### GDI annotations
    #### we print performance for top k speaking characters
    #### k = inf means take all characters
    #####################################################################
    

    k_list = [5, 10, 20, np.inf]

    for k in k_list:

        tp, fp, fn = 0, 0, 0
        errors = []

        if np.isinf(k):
            print("all characters")
        else:
            print("top {} speaking characters".format(k))
        
        for folder, movie in items:

            gitems = []
            for _, row in line_count_df.loc[(line_count_df["folder"] == folder) & (line_count_df["movie"] == movie), ["character", "line-count"]].iterrows():
                gitems.append((row["character"], int(row["line-count"])))
            gitems = sorted(gitems, key=lambda citem: citem[1], reverse=True)
            if not np.isinf(k):
                gitems = gitems[:k]
            cdict = {}
            for ch, cn in gitems:
                cdict[ch] = cn

            citems = []
            for _, row in sys_line_count_df.loc[(sys_line_count_df["folder"] == folder) & (sys_line_count_df["movie"] == movie), ["character", "line-count"]].iterrows():
                citems.append((row["character"], int(row["line-count"])))
            citems = sorted(citems, key=lambda citem: citem[1], reverse=True)
            if not np.isinf(k):
                citems = citems[:k]
            sys_cdict = {}
            for ch, cn in citems:
                sys_cdict[ch] = cn
            
            characters = set(list(cdict.keys()))
            sys_characters = set(list(sys_cdict.keys()))
            common_characters = characters.intersection(sys_characters)

            _tp = len(common_characters)
            _fp = len(sys_characters.difference(characters))
            _fn = len(characters.difference(sys_characters))
            _errors = []
            for character in common_characters:
                _errors.append(sys_cdict[character] - cdict[character])

            _p = _tp/(_tp + _fp)
            _r = _tp/(_tp + _fn)
            _f1 = 2 * _p * _r / (_p + _r + 1e-23)
            _mean = np.mean(_errors)
            _std = np.std(_errors)
            _median = np.median(_errors)

            print("\tcharacter: p = {:5.2f}, r = {:5.2f}, f1 = {:5.2f} ; line count error: mean = {:5.1f}, std = {:5.1f}, median = {:4.1f} ; {}/{}".format(_p, _r, _f1, _mean, _std, _median, folder, movie))

            tp += _tp
            fp += _fp
            fn += _fn
            errors.extend(_errors)            
        
        print()

        p = tp/(tp + fp)
        r = tp/(tp + fn)
        f1 = 2 * p * r / (p + r)
        mean = np.mean(errors)
        std = np.std(errors)
        median = np.median(errors)

        print("character: p = {:5.2f}, r = {:5.2f}, f1 = {:5.2f} ; line count error: mean = {:5.1f}, std = {:5.1f}, median = {:4.1f} ; ALL".format(p, r, f1, mean, std, median))