# author : Sabyasachee

# standard library
from collections import Counter
import os
import re

# third party
from docx import Document
import numpy as np
import pandas as pd

def evaluate_gdi(gdi_folder, gdi_folder_names, ignore_scripts=[]):

    #####################################################################
    #### get GDI annotated character line counts
    #####################################################################

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
                        data.append([folder, movie, cells[i].strip().lower(), cells[i + 1]])
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
    
    data = []

    for folder, movie in items:
        slines = open(os.path.join(gdi_folder, "{}/Scripts_Txt/{}.txt".format(folder, movie))).read().splitlines()
        tags = open(os.path.join(gdi_folder, "{}/Parsed/{}_tags.txt".format(folder, movie))).read().splitlines()
        tags = [t if t in ["S","N","C","D","E","T"] else "O" for t in tags]

        characters = []
        n = len(slines)
        i = 0

        while i < n:
            if tags[i] == "C":
                character = re.sub("(^[\s\d\*]*)|((\(.+\)\s*)*[\s\d\*]*$)", "", slines[i]).strip().lower()

                if len(character) > 2:
                    i += 1
                    while i < n and tags[i] in ["0","D","E"]:
                        if tags[i] == "D":
                            characters.append(character)
                        i += 1
                    i -= 1
            i += 1

        citems = sorted(Counter(characters).items(), key = lambda x: x[1], reverse=True)
        for ch, cn in citems:
            data.append([folder, movie, ch, cn])

    sys_line_count_df = pd.DataFrame(data, columns=["folder", "movie", "character", "line-count"])
    sys_line_count_df.to_csv(os.path.join(gdi_folder, "parser_line_counts.tsv"), sep="\t", index=False)

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
            _mae = np.mean(np.absolute(_errors))
            _rmse = np.sqrt(np.mean(np.square(_errors)))
            _median = np.median(np.absolute(_errors))

            # print("\tcharacter: p = {:5.2f}, r = {:5.2f}, f1 = {:5.2f} ; line count error: mae = {:5.1f}, rmse = {:5.1f}, median = {:4.1f} ; {}/{}".format(_p, _r, _f1, _mae, _rmse, _median, folder, movie))

            tp += _tp
            fp += _fp
            fn += _fn
            errors.extend(_errors)            
        
        print()

        p = tp/(tp + fp)
        r = tp/(tp + fn)
        f1 = 2 * p * r / (p + r)
        mae = np.mean(np.absolute(errors))
        rmse = np.sqrt(np.mean(np.square(errors)))
        median = np.median(np.absolute(errors))

        print("character: p = {:5.2f}, r = {:5.2f}, f1 = {:5.2f} ; line count error: mae = {:5.1f}, rmse = {:5.1f}, median = {:4.1f} ; ALL".format(p, r, f1, mae, rmse, median))