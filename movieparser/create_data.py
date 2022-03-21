# author : Sabyasachee

# standard library
from collections import defaultdict
import os
import random
import re
from typing import List, Tuple

# third-party library
import pandas as pd

# user library
from movieparser.parse_scripts_noindent import parse_lines

def replace_name_with_name_containing_keyword(script, label, keywords, names_file):
    script = script[:]
    names = open(names_file).read().splitlines()
    names = sorted(set([name.upper() for name in names if any(keyword in name.lower() for keyword in keywords)]))
    random.shuffle(names)

    cdata = defaultdict(list)

    for i, (line, tag) in enumerate(zip(script, label)):
        if tag == "C":
            cdata[line.strip()].append(i)
    
    characters = sorted(cdata.keys(), key = lambda character: len(cdata[character]), reverse=True)

    n_replacements = min(len(names), len(characters))

    for name, character in zip(names[:n_replacements], characters[:n_replacements]):
        for i in cdata[character]:
            script[i] = name
    
    return script, label

def replace_name_with_name_containing_scene_keyword(script, label, names_file):
    keywords = ["int", "ext"]
    return replace_name_with_name_containing_keyword(script, label, keywords, names_file)

def replace_name_with_name_containing_transition_keyword(script, label, names_file):
    keywords = ["cut","fade"]
    return replace_name_with_name_containing_keyword(script, label, keywords, names_file)

def remove_scene_keyword_from_scene_headers(script, label):
    script = script[:]
    script = [re.sub("(IN|EX)T\.?", " ", line).strip() if tag == "S" else line for line, tag in zip(script, label)]
    return script, label

def lowercase_scene_headers(script, label):
    script = script[:]
    script = [line.lower() if tag == "S" else line for line, tag in zip(script, label)]
    return script, label

def lowercase_character_names(script, label):
    script = script[:]
    script = [line.lower() if tag == "C" else line for line, tag in zip(script, label)]
    return script, label

def insert_watermark_lines(script, label, prob=0.2):
    watermarked_script = []
    watermarked_label = []
    characters = [chr(i) for i in range(65,91)]

    for line, tag in zip(script, label):
        watermarked_script.append(line)
        watermarked_label.append(tag)

        if random.random() < prob:
            watermark = "".join([random.choice(characters) for _ in range(random.choice([1,2,3]))])
            watermarked_script.append(watermark)
            watermarked_label.append("M")

    return watermarked_script, watermarked_label

def insert_asterisks_or_numbers(script, label, prob=0.5):
    script = script[:]

    for i, line in enumerate(script):
        if random.random() < prob:
            leading_character = "*" if random.random() < 0.25 else str(random.choice(list(range(100))))
            trailing_character = "*" if random.random() < 0.25 else str(random.choice(list(range(100))))
            script[i] = leading_character + "     " + line + "     " + trailing_character

    return script, label

def insert_dialogue_expressions(script, label, prob=0.1):
    expressions = ["beat", "pause", "smiling", "continuing", "contd.", "quietly", "shouting", "yelling", \
            "grinning", "screaming", "anxious", "interrupting"]

    new_script, new_label = [], []

    for line, tag in zip(script, label):
        new_script.append(line)
        new_label.append(tag)
        if tag == "D" and random.random() < prob:
            new_script.append("({})".format(random.choice(expressions)))
            new_label.append("E")

    return new_script, new_label

def create_data(annotator_1_file, annotator_2_file, annotator_3_file, line_indices_file, names_file="data/names.txt", results_folder="results", screenplays_folder="data/SAIL_annotation_screenplays/screenplays"):

    random.seed(0)

    #####################################################################
    #### read annotator excel files
    #####################################################################

    print("reading annotator 1 file...")
    annotator_1_df_dict = pd.read_excel(annotator_1_file, sheet_name=None, header=1, \
                                        usecols=["line","S","N","C","D","E","T","M"])

    print("reading annotator 2 file...")
    annotator_2_df_dict = pd.read_excel(annotator_2_file, sheet_name=None, header=1, \
                                        usecols=["line","S","N","C","D","E","T","M"])

    print("reading annotator 3 file...")
    annotator_3_df_dict = pd.read_excel(annotator_3_file, sheet_name=None, header=1, \
                                        usecols=["line","S","N","C","D","E","T","M"])

    line_indices = open(line_indices_file).read().splitlines()

    #####################################################################
    #### create movie data
    #####################################################################
    
    data = {}
    print("collecting annotations...")

    for line in line_indices:
        movie, _, start, end = line.split()
        start = int(start)
        end = int(end)

        script_file = os.path.join(screenplays_folder, movie + ".txt")
        script = open(script_file).read().splitlines()

        pre_script = [line.strip() for line in script[:start] if len(line.strip()) > 0]
        in_script = [line.strip() for line in script[start:end] if len(line.strip()) > 0]
        post_script = [line.strip() for line in script[end:] if len(line.strip()) > 0]

        start = len(pre_script)
        end = start + len(in_script)

        all_anns = []

        for df in [annotator_1_df_dict[movie], annotator_2_df_dict[movie], annotator_3_df_dict[movie]]:
            anns = []

            for _, row in df.iterrows():
                if pd.notna(row["line"]) and str(row["line"]).strip() != "":
                    n = 0
                    atag = ""
                    for tag in ["S","N","C","D","E","T","M"]:
                        if str(row["line"]).strip() == str(row[tag]).strip():
                            atag = tag
                        else:
                            n += 1
                    if atag != "" and n == 6:
                        anns.append(atag)
                    else:
                        anns.append("O")
            
            all_anns.append(anns)
        
        maj = []

        for a1, a2, a3 in zip(all_anns[0], all_anns[1], all_anns[2]):
            if a1 == a2 or a1 == a3:
                maj.append(a1)
            elif a2 == a3:
                maj.append(a2)
            else:
                maj.append("O")
        
        parse = parse_lines(in_script)

        data[movie] = {
            "start": start,
            "end": end,
            "full_script": pre_script + in_script + post_script,
            "script": in_script,
            "label": maj,
            "parse": parse
        }

    #####################################################################
    #### create training data for each error, and save it to results
    #####################################################################
    
    records = []
    header = ["movie", "line_no", "text", "label", "error"]
    print("creating training data...")

    for movie, mdata in data.items():

        for i, (line, tag) in enumerate(zip(mdata["script"], mdata["label"])):
            records.append([movie, i + 1, line, tag, "NONE"])
        
        script, label = replace_name_with_name_containing_scene_keyword(mdata["script"], mdata["label"], names_file)
        for i, (line, tag) in enumerate(zip(script, label)):
            records.append([movie, i + 1, line, tag, "REPLACE_NAME_SCENE"])
        
        script, label = replace_name_with_name_containing_transition_keyword(mdata["script"], mdata["label"], names_file)
        for i, (line, tag) in enumerate(zip(script, label)):
            records.append([movie, i + 1, line, tag, "REPLACE_NAME_TRANSITION"])
        
        script, label = remove_scene_keyword_from_scene_headers(mdata["script"], mdata["label"])
        for i, (line, tag) in enumerate(zip(script, label)):
            records.append([movie, i + 1, line, tag, "REMOVE_SCENE_KEYWORD"])
        
        script, label = lowercase_scene_headers(mdata["script"], mdata["label"])
        for i, (line, tag) in enumerate(zip(script, label)):
            records.append([movie, i + 1, line, tag, "LOWERCASE_SCENE_HEADER"])

        script, label = lowercase_character_names(mdata["script"], mdata["label"])
        for i, (line, tag) in enumerate(zip(script, label)):
            records.append([movie, i + 1, line, tag, "LOWERCASE_CHARACTER_NAME"])
        
        script, label = insert_watermark_lines(mdata["script"], mdata["label"])
        for i, (line, tag) in enumerate(zip(script, label)):
            records.append([movie, i + 1, line, tag, "WATERMARK"])
        
        script, label = insert_asterisks_or_numbers(mdata["script"], mdata["label"])
        for i, (line, tag) in enumerate(zip(script, label)):
            records.append([movie, i + 1, line, tag, "ASTERISKS_NUMBERS"])

        script, label = insert_dialogue_expressions(mdata["script"], mdata["label"])
        for i, (line, tag) in enumerate(zip(script, label)):
            records.append([movie, i + 1, line, tag, "DIALOGUE_EXPRESSIONS"])
    
    df = pd.DataFrame(records, columns=header)
    df_file = os.path.join(results_folder, "data.csv")
    df.to_csv(df_file, index=False)