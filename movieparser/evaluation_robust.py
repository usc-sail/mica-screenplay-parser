# author : Sabyasachee

# standard library
import os
import re
import random
from typing import List, Tuple

# third library
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from tqdm import trange, tqdm

# user library
from movieparser.parse_scripts_noindent import parse_lines

class RobustEvaluation:

    def err_remove_blank_lines_and_indents(self, script: List[str], start: int, end: int, tags: List[str]) \
        -> Tuple[List[str], int, int, List[str]]:
        
        new_script, new_tags = [], []
        new_start, new_end = 0, 0

        for i, line in enumerate(script):

            if i == start:
                new_start = len(new_script)
            elif i == end:
                new_end = len(new_script)
            
            if i < start or i >= end:
                if len(line.strip()) > 0:
                    new_script.append(line.strip())
            else:
                if len(line.strip()) > 0:
                    new_script.append(line.strip())
                    new_tags.append(tags[i - start])
        
        return new_script, new_start, new_end, new_tags

    def err_replace_name_with_name_containing_scene_keyword(self, script: List[str], start: int, end: int, \
        tags: List[str], prob: float, names_file: str, **kwargs) -> Tuple[List[str], int, int, List[str]]:

        names = open(names_file).read().splitlines()
        keywords = ["int", "ext"]
        names = [name.upper() for name in names if any(keyword in name.lower() for keyword in keywords)]

        new_script = script[:]

        for i, line in enumerate(script[start: end]):
            if tags[i] == "C" and re.match("^[a-zA-Z]+$", line.strip()) and random.random() < prob:
                new_script[start + i] = random.choice(names)
        
        return new_script, start, end, tags

    def err_replace_name_with_name_containing_transition_keyword(self, script: List[str], start: int, end: int, \
        tags: List[str], prob: float, names_file: str, **kwargs) -> Tuple[List[str], int, int, List[str]]:

        names = open(names_file).read().splitlines()
        keywords = ["cut", "fade"]
        names = [name.upper() for name in names if any(keyword in name.lower() for keyword in keywords)]

        new_script = script[:]

        for i, line in enumerate(script[start: end]):
            if tags[i] == "C" and re.match("^[a-zA-Z]+$", line.strip()) and random.random() < prob:
                new_script[start + i] = random.choice(names)
        
        return new_script, start, end, tags

    def err_remove_scene_keyword(self, script: List[str], start: int, end: int, tags: List[str], prob: float, \
        **kwargs) -> Tuple[List[str], int, int, List[str]]:

        new_script = script[:]

        for i, line in enumerate(script[start: end]):
            if tags[i] == "S" and random.random() < prob:
                new_script[start + i] = re.sub("(IN|EX)T\.?", " ", line).strip()

        return new_script, start, end, tags

    def err_lowercase_scene_line(self, script: List[str], start: int, end: int, tags: List[str], prob: float, \
        **kwargs) -> Tuple[List[str], int, int, List[str]]:

        new_script = script[:]

        for i, line in enumerate(script[start: end]):
            if tags[i] == "S" and random.random() < prob:
                new_script[start + i] = line.lower()

        return new_script, start, end, tags
    
    def err_lowercase_character_line(self, script: List[str], start: int, end: int, tags: List[str], prob: float, \
        **kwargs) -> Tuple[List[str], int, int, List[str]]:

        new_script = script[:]

        for i, line in enumerate(script[start: end]):
            if tags[i] == "C" and random.random() < prob:
                new_script[start + i] = line.lower()

        return new_script, start, end, tags

    def err_create_watermark_lines(self, script: List[str], start: int, end: int, tags: List[str], prob: float, \
        **kwargs) -> Tuple[List[str], int, int, List[str]]:

        new_script, new_tags = [], []
        new_start, new_end = 0, 0
        characters = [chr(i) for i in range(65,91)]

        for i, line in enumerate(script):

            if i == start:
                new_start = len(new_script)
            elif i == end:
                new_end = len(new_script)

            new_script.append(line)
            if start <= i < end:
                new_tags.append(tags[i - start])

            if random.random() < prob:
                watermark = "".join([random.choice(characters) for _ in range(random.choice([1,2,3]))])
                if start <= i < end:
                    new_script.append(watermark)
                    new_tags.append("M")
                else:
                    new_script.append(watermark)

        return new_script, new_start, new_end, new_tags

    def err_insert_asterisks_or_numbers(self, script: List[str], start: int, end: int, tags: List[str], prob: float, \
         **kwargs) -> Tuple[List[str], int, int, List[str]]:

        new_script = script[:]

        for i, line in enumerate(script[start: end]):
            if random.random() < prob:
                leading_character = "*" if random.random() < 0.25 else str(random.choice(list(range(100))))
                trailing_character = "*" if random.random() < 0.25 else str(random.choice(list(range(100))))
                new_script[start + i] = leading_character + "     " + line + "     " + trailing_character

        return new_script, start, end, tags

    def err_insert_dialogue_expressions(self, script: List[str], start: int, end: int, tags: List[str], prob: float, \
        **kwargs) -> Tuple[List[str], int, int, List[str]]:

        expressions = ["beat", "pause", "smiling", "continuing", "contd.", "quietly", "shouting", "yelling", \
            "grinning", "screaming", "anxious", "interrupting"]

        new_script, new_tags = [], []
        new_start, new_end = 0, 0

        for i, line in enumerate(script):

            if i == start:
                new_start = len(new_script)
            elif i == end:
                new_end = len(new_script)
                
            if start <= i < end and tags[i - start] == "D":
                new_script.append("({})".format(random.choice(expressions)))
                new_tags.append("E")
            
            new_script.append(line)
            if start <= i < end:
                new_tags.append(tags[i - start])

        return new_script, new_start, new_end, new_tags

    def __init__(self,
                annotator_1_file, annotator_2_file, annotator_3_file, 
                screenplays_folder, 
                screenplays_line_numbers_file,
                names_file="data/names.txt",
                results_folder="results/"
                ):

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

        line_numbers = open(screenplays_line_numbers_file).read().splitlines()

        #####################################################################
        #### find annotations of each rater and save them to "ann" dictionary
        #### find majority
        #####################################################################

        movies = list(annotator_1_df_dict.keys())[1:]
        ann = {}

        for movie in movies:
            ann[movie] = {}
            
            for i, df in enumerate([annotator_1_df_dict[movie], annotator_2_df_dict[movie], annotator_3_df_dict[movie]]):
                anns = []

                for _, row in df.iterrows():
                    if not isinstance(row["line"], str) or (isinstance(row["line"], str) and row["line"].strip() == ""):
                        anns.append("O")
                    else:
                        n = 0
                        atag = ""
                        for tag in ["S","N","C","D","E","T","M"]:
                            if isinstance(row[tag], str) and row["line"].strip() == row[tag].strip():
                                atag = tag
                            else:
                                n += 1
                        if atag != "" and n == 6:
                            anns.append(atag)
                        else:
                            anns.append("O")
                
                ann[movie]["ann{}".format(i + 1)] = anns
            
            maj = []
            for a1, a2, a3 in zip(ann[movie]["ann1"], ann[movie]["ann2"], ann[movie]["ann3"]):
                if a1 == a2 or a1 == a3:
                    maj.append(a1)
                elif a2 == a3:
                    maj.append(a2)
                else:
                    maj.append("O")
            ann[movie]["gold"] = maj

        #####################################################################
        #### parse script
        #####################################################################
        
        for line in line_numbers:
            movie, _, i, j = line.split()
            i, j = int(i), int(j)
            script = open(os.path.join(screenplays_folder, movie + ".txt")).read().splitlines()
            tags = parse_lines(script)
            tags = [t if t in ["S","N","C","D","E","T","M"] else "O" for t in tags]
            ann[movie]["sys"] = tags[i:j]
            ann[movie]["start"] = i
            ann[movie]["end"] = j
            ann[movie]["script"] = script

        self.ann = ann
        self.names_file = names_file
        self.results_folder = results_folder

    def evaluate(self, gold_key, sys_key):

        f1_dict = {}
        p_dict = {}
        r_dict = {}

        for tag in ["S","N","C","D","E","T"]:
            tp, fp, fn = 0, 0, 0
            for movie in self.ann.keys():
                tp += sum(x == y == tag for x, y in zip(self.ann[movie][gold_key], self.ann[movie][sys_key]))
                fp += sum(x != y == tag for x, y in zip(self.ann[movie][gold_key], self.ann[movie][sys_key]))
                fn += sum(tag == x != y for x, y in zip(self.ann[movie][gold_key], self.ann[movie][sys_key]))
            p = tp/(tp + fp + 1e-23)
            r = tp/(tp + fn + 1e-23)
            f1 = 2*p*r/(p + r + 1e-23)
            
            p_dict[tag] = p
            r_dict[tag] = r
            f1_dict[tag] = f1
        
        return p_dict, r_dict, f1_dict
    
    def robust_evaluate(self):

        #####################################################################
        #### create dictionaries and arrays to store performance data
        #####################################################################

        p_dict, r_dict, f1_dict = {}, {}, {}
        names = []
        types = []
        probs = []
        
        #####################################################################
        #### evaluation on original script
        #####################################################################
        
        _p_dict, _r_dict, _f1_dict = self.evaluate("gold", "sys")
        names.append("no error")
        types.append("original")
        probs.append(None)
        for tag in ["S","N","C","D","E","T"]:
            p_dict[tag] = [_p_dict[tag]]
            r_dict[tag] = [_r_dict[tag]]
            f1_dict[tag] = [_f1_dict[tag]]

        #####################################################################
        #### evaluation on contiguous script - blank lines and indents
        #### removed
        #####################################################################
        
        for movie in self.ann:
            script, start, end, tags = self.ann[movie]["script"], self.ann[movie]["start"], \
                                        self.ann[movie]["end"], self.ann[movie]["gold"]
            contiguous_script, contiguous_start, contiguous_end, contiguous_tags = \
                self.err_remove_blank_lines_and_indents(script, start, end, tags)
            self.ann[movie]["cgold"] = contiguous_tags
            self.ann[movie]["csys"] = parse_lines(contiguous_script)[contiguous_start: contiguous_end]
            self.ann[movie]["cscript"] = contiguous_script
            self.ann[movie]["cstart"] = contiguous_start
            self.ann[movie]["cend"] = contiguous_end
        
        _p_dict, _r_dict, _f1_dict = self.evaluate("cgold", "csys")
        names.append("no error")
        probs.append(None)
        types.append("contiguous")
        for tag in ["S","N","C","D","E","T"]:
            p_dict[tag].append(_p_dict[tag])
            r_dict[tag].append(_r_dict[tag])
            f1_dict[tag].append(_f1_dict[tag])

        #####################################################################
        #### types of errors and their functions
        #####################################################################
        
        errors = ["replace_name_with_scene_kw",
                    "replace_name_with_transition_kw",
                    "remove_scene_kw",
                    "lowercase_scene_line",
                    "lowercase_character_line",
                    "create_watermark_line",
                    "insert_asterisks_or_numbers",
                    "insert_dialogue_expressions"
                    ]
        functions = [self.err_replace_name_with_name_containing_scene_keyword,
                    self.err_replace_name_with_name_containing_transition_keyword,
                    self.err_remove_scene_keyword,
                    self.err_lowercase_scene_line,
                    self.err_lowercase_character_line,
                    self.err_create_watermark_lines,
                    self.err_insert_asterisks_or_numbers,
                    self.err_insert_dialogue_expressions
                    ]

        #####################################################################
        #### iterate over errors
        #####################################################################

        for error, function in zip(errors, functions):
            print(error)

            #####################################################################
            #### iterate over error/line probabilities
            #####################################################################
            
            for p in [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                print("\tprob = ", p)

                trials_p_dict, trials_r_dict, trials_f1_dict = {}, {}, {}
                trials_p_dict1, trials_r_dict1, trials_f1_dict1 = {}, {}, {}
                
                for tag in ["S","N","C","D","E","T"]:
                    trials_p_dict[tag] = []
                    trials_p_dict1[tag] = []
                    trials_r_dict[tag] = []
                    trials_r_dict1[tag] = []
                    trials_f1_dict[tag] = []
                    trials_f1_dict1[tag] = []

                #####################################################################
                #### iterate over trials. final error will be averaged
                #####################################################################
                
                print("\t\ttrials = ", end="")

                for i in range(10):
                    print(i + 1, end=" ")

                    #####################################################################
                    #### iterate over movies
                    #####################################################################
                    
                    for movie in self.ann:
                        new_script, new_start, new_end, new_tags = function(self.ann[movie]["script"], self.ann[movie]["start"], self.ann[movie]["end"], self.ann[movie]["gold"], p, names_file=self.names_file)
                        self.ann[movie]["rgold"] = new_tags[:]
                        self.ann[movie]["rsys"] = parse_lines(new_script)[new_start: new_end][:]

                        new_script, new_start, new_end, new_tags = function(self.ann[movie]["cscript"], self.ann[movie]["cstart"], self.ann[movie]["cend"], self.ann[movie]["cgold"], p, names_file=self.names_file)
                        self.ann[movie]["rcgold"] = new_tags[:]
                        self.ann[movie]["rcsys"] = parse_lines(new_script)[new_start: new_end][:]
                
                    _p_dict, _r_dict, _f1_dict = self.evaluate("rgold", "rsys")
                    _p_dict1, _r_dict1, _f1_dict1 = self.evaluate("rcgold", "rcsys")

                    for tag in ["S","N","C","D","E","T"]:
                        trials_p_dict[tag].append(_p_dict[tag])
                        trials_r_dict[tag].append(_r_dict[tag])
                        trials_f1_dict[tag].append(_f1_dict[tag])
                        trials_p_dict1[tag].append(_p_dict1[tag])
                        trials_r_dict1[tag].append(_r_dict1[tag])
                        trials_f1_dict1[tag].append(_f1_dict1[tag])
                
                print()
                
                #####################################################################
                #### take average of trials
                #####################################################################
                
                names.append(error)
                probs.append(p)
                types.append("original")
                for tag in ["S","N","C","D","E","T"]:
                    p_dict[tag].append(np.mean(trials_p_dict[tag]))
                    r_dict[tag].append(np.mean(trials_r_dict[tag]))
                    f1_dict[tag].append(np.mean(trials_f1_dict[tag]))
                
                names.append(error)
                probs.append(p)
                types.append("contiguous")
                for tag in ["S","N","C","D","E","T"]:
                    p_dict[tag].append(np.mean(trials_p_dict1[tag]))
                    r_dict[tag].append(np.mean(trials_r_dict1[tag]))
                    f1_dict[tag].append(np.mean(trials_f1_dict1[tag]))
        
        #####################################################################
        #### create three dataframes for precision, recall, and f1
        #####################################################################
        
        precision_df = pd.DataFrame.from_dict(p_dict)
        recall_df = pd.DataFrame.from_dict(r_dict)
        f1_df = pd.DataFrame.from_dict(f1_dict)

        precision_df["error"] = recall_df["error"] = f1_df["error"] = names
        precision_df["prob"] = recall_df["prob"] = f1_df["prob"] = probs
        precision_df["type"] = recall_df["type"] = f1_df["type"] = types

        results_folder = os.path.join(self.results_folder, "robust_evaluation")
        os.makedirs(results_folder, exist_ok=True)

        precision_df.to_csv(os.path.join(results_folder, "precision.csv"), index=False)
        recall_df.to_csv(os.path.join(results_folder, "recall.csv"), index=False)
        f1_df.to_csv(os.path.join(results_folder, "f1.csv"), index=False)

        #####################################################################
        #### create 9x9 subplots for each tag
        #####################################################################

        for tag in ["S","N","C","D","T"]:

            os.makedirs(os.path.join(results_folder, tag), exist_ok=True)

            for metric, df in zip(["F1", "precision", "recall"], [f1_df, precision_df, recall_df]):

                original = df.loc[(df.error == "no error") & (df.type == "original"), tag].values[0]
                contiguous = df.loc[(df.error == "no error") & (df.type == "contiguous"), tag].values[0]

                fig, axs = plt.subplots(3, 3)

                fig.set_figheight(20)
                fig.set_figwidth(20)

                for i, error_type in enumerate(errors):
                    
                    original_arr = df.loc[(df.error == error_type) & (df.type == "original"), [tag, "prob"]]
                    contiguous_arr = df.loc[(df.error == error_type) & (df.type == "contiguous"), [tag, "prob"]]

                    r = (i + 1)//3
                    c = (i + 1)%3
                    ax = axs[r, c]

                    ax.plot(original_arr.prob, original_arr[tag], color="b", label="original", lw=4)
                    ax.plot(contiguous_arr.prob, contiguous_arr[tag], color="r", label="contiguous", lw=4)
                    title = error_type.replace("_", " ").upper()
                    ax.set_title(title)
                    ax.set_xlim(-0.1, 1)
                    ax.set_ylim(-0.1, 1)

                    ax.set_xlabel("error/line")
                    ax.set_ylabel(metric)

                    ax.legend()

                ax = axs[0, 0]
                ax.bar(["original", "contiguous"], [original, contiguous], color=["b","r"])

                fig.savefig(os.path.join(results_folder, "{}/{}-{}.png".format(tag, tag, metric)))
                plt.close("all")