"""Find features for each screenplay line of the MovieParse dataset.
"""
import os
import re
from typing import List, Tuple
import pandas as pd
import spacy
from spacy.tokens.doc import Doc
from tqdm import tqdm

def get_pos_features(doc: Doc) -> Tuple[List[float], List[str]]:
    feature = []
    header = ["num_noun", "frac_noun", "num_verb", "frac_noun", "num_adjective", "frac_adjective", "num_adverb", 
              "frac_adverb"]
    for pos_prefix in ["NN","VB","JJ","RB"]:
        n_pos_words = sum(token.tag_.startswith(pos_prefix) for token in doc)
        if len(doc):
            feature.extend([n_pos_words, n_pos_words/len(doc)])
        else:
            feature.extend([0., 0.])
    return feature, header

def get_entity_features(doc: Doc) -> Tuple[List[float], List[str]]:
    feature = []
    header = ["num_person", "frac_person", "num_geopolitical", "frac_geopolitical", "num_location", "frac_location", 
              "num_organization", "frac_organization", "num_time", "frac_time", "num_date", "frac_date"]
    for label in ["PERSON","GPE","LOC","ORG","TIME","DATE"]:
        n_ent_words = sum(token.ent_type_ == label for token in doc)
        if len(doc):
            feature.extend([n_ent_words, n_ent_words/len(doc)])
        else:
            feature.extend([0., 0.])
    return feature, header

def get_length_features(doc: Doc) -> Tuple[List[float], List[str]]:
    feature = [len(doc)]
    header = ["length"]
    return feature, header

def get_capitalization_features(doc: Doc) -> Tuple[List[float], List[str]]:
    n_cap_words = sum(token.is_upper for token in doc)
    if len(doc):
        feature = [n_cap_words, n_cap_words/len(doc)]
    else:
        feature = [0., 0.]
    header = ["num_cap", "frac_cap"]
    return feature, header

def get_parentheses_features(doc: Doc) -> Tuple[List[float], List[str]]:
    n_left = sum(ch == "(" for ch in doc.text)
    n_right = sum(ch == ")" for ch in doc.text)
    feature = [n_left, n_right, n_left - n_right]
    header = ["num_left_parentheses", "n_right_parentheses", "n_open_parentheses"]
    return feature, header

def get_keyphrase_features(doc: Doc) -> Tuple[List[float], List[str]]:
    uncased_transition_keyphrases = ["cut to", "cut back to", "transition to", "close on", "dissolve to", 
                                     "shock cut to", "fade in", "fade up", "fade to", "fade out"]
    uncased_scene_keyphrases = ["int", "ext"]
    uncased_keyphrases = uncased_transition_keyphrases + uncased_scene_keyphrases
    feature = []
    header = []
    for keyphrase in uncased_keyphrases:
        feature.append(int(re.search("(\A|\W)" + re.escape(keyphrase) + "(\W|\Z)", doc.text.lower()) is not None))
        header.append("contains_" + keyphrase.replace(" ", "_"))
    return feature, header

def create_features(results_folder:str):
    df = pd.read_csv(os.path.join(results_folder, "data.csv"), index_col=None)
    sentences = df["text"].fillna("").unique()
    print("{} unique sentences".format(len(sentences)))

    spacy.require_gpu()
    nlp = spacy.load("en_core_web_lg", disable=["parser"])
    
    data = []
    header = []

    feature_functions = [get_pos_features, get_entity_features, get_length_features, get_capitalization_features, 
                         get_parentheses_features, get_keyphrase_features]
    
    for doc in tqdm(nlp.pipe(sentences, batch_size=1024), total=len(sentences), desc="feature extraction"):
        header = []
        vector = []
        for function in feature_functions:
            feature, feature_header = function(doc)
            vector.extend(feature)
            header.extend(feature_header)
        data.append([doc.text] + vector)
    
    df = pd.DataFrame(data, columns=["text"] + header)
    df.to_csv(os.path.join(results_folder, "feats.csv"), index=False)

class FeatureExtractor:

    def __init__(self, gpu_id=0) -> None:
        spacy.require_gpu(gpu_id)
        self.nlp = spacy.load("en_core_web_lg", disable=["parser"])
    
    def __call__(self, sentences: List[str]) -> List[List[float]]:
        feature_functions = [get_pos_features, get_entity_features, get_length_features, get_capitalization_features, 
                             get_parentheses_features, get_keyphrase_features]
        vectors = []
        for doc in self.nlp.pipe(sentences, batch_size=1024):
            vector = []
            for function in feature_functions:
                feature, _ = function(doc)
                vector.extend(feature)
            vectors.append(vector)
        return vectors
