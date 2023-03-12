# author : Sabyasachee

# standard library imports
from collections import defaultdict
import math
import os
from typing import Union, Tuple, List

# third party imports
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from torch import nn
import torch

# user library imports
from mica_text_parser.screenplayparser.feats import FeatureExtractor

class RobustScreenplayParser(nn.Module):

    def __init__(self, n_features: int, n_labels: int, bidirectional: bool, features_file="results/feats.csv", device_index: int = 0) -> None:
        super().__init__()
        label2id = defaultdict(int)
        for i, label in enumerate("OSNCDETM"):
            label2id[label] = i
        self.encoder = SentenceTransformer("all-mpnet-base-v2")
        self.feature_size = self.encoder.get_sentence_embedding_dimension() + n_features
        self.hidden_size = 256
        self.n_labels = n_labels
        self.bidirectional = bidirectional

        self.features_cache = {}
        if os.path.exists(features_file):
            df = pd.read_csv(features_file, index_col=0)
            for text, feature in df.iterrows():
                self.features_cache[text] = feature.values
        self.feature_extractor = FeatureExtractor(gpu_id=device_index)

        self.id2label = dict((i, label) for label, i in label2id.items())
        
        self.lstm = nn.LSTM(self.feature_size, self.hidden_size, batch_first=True, bidirectional=bidirectional)
        self.classifier = nn.Linear((1 + int(self.bidirectional)) * self.hidden_size, self.n_labels)
    
    def forward(self, scripts: np.ndarray, features: torch.FloatTensor, labels: torch.LongTensor = None) -> Union[torch.LongTensor, Tuple[torch.Tensor, torch.LongTensor]]:
        batch_size, seqlen = scripts.shape
        device = next(self.parameters()).device
        script_embeddings = self.encoder.encode(scripts.flatten(), convert_to_tensor=True, device=device).reshape(batch_size, seqlen, -1)
        input = torch.cat([script_embeddings, features], dim=2)
        output, _ = self.lstm(input)
        logits = self.classifier(output)
        pred = logits.argmax(dim=2)
        if labels is None:
            return pred
        else:
            flabels = labels.flatten().tolist()
            flabels.extend(list(range(self.n_labels)))
            class_distribution = np.bincount(flabels)
            class_weights = 1/class_distribution
            normalized_class_weights = class_weights/class_weights.sum()
            normalized_class_weights_tensor = torch.FloatTensor(normalized_class_weights).to(device)
            ce_loss = nn.CrossEntropyLoss(normalized_class_weights_tensor)
            loss = ce_loss(logits.reshape(-1, self.n_labels), labels.flatten())
            return loss, pred
    
    def parse(self, script: List[str], segment_length: int = 1000000) -> List[str]:
        
        if segment_length is None:
            segment_length = len(script)
        n_segments = math.ceil(len(script)/segment_length)
        device = next(self.parameters()).device
        
        features = self.feature_extractor(script)
        features = torch.FloatTensor(features).to(device)

        pred = []

        for i in range(n_segments):
            segment = script[i * segment_length: (i + 1) * segment_length]
            segment_features = features[i * segment_length: (i + 1) * segment_length]
            segment_embeddings = self.encoder.encode(segment, convert_to_tensor=True, device=device)
            input = torch.cat([segment_embeddings, segment_features], dim=1)
            input = torch.unsqueeze(input, dim=0)
            output, _ = self.lstm(input)
            output = torch.squeeze(output, dim=0)
            logits = self.classifier(output)
            _pred = logits.argmax(dim=1)
            pred.extend(_pred.cpu().tolist())
        
        pred = [self.id2label[p] for p in pred]
        return pred