# author : Sabyasachee

# standard library imports
import math
from typing import Union, Tuple, List

# third party imports
import numpy as np
from torch import nn
import torch
from sentence_transformers import SentenceTransformer

class ScriptParser(nn.Module):

    def __init__(self, n_features: int, n_labels: int) -> None:
        super().__init__()
        self.encoder = SentenceTransformer("all-mpnet-base-v2")
        self.feature_size = self.encoder.get_sentence_embedding_dimension() + n_features
        self.hidden_size = 256
        self.n_labels = n_labels
        self.lstm = nn.LSTM(self.feature_size, self.hidden_size, batch_first=True)
        self.classifier = nn.Linear(self.hidden_size, self.n_labels)
    
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
    
    def parse(self, script: List[str], features: torch.FloatTensor) -> List[int]:
        segment_length = 10
        n_segments = math.ceil(len(script)/segment_length)
        device = next(self.parameters()).device
        pred = []
        hidden, cell = torch.zeros((1, self.hidden_size), device=device), torch.zeros((1, self.hidden_size), device=device)

        for i in range(n_segments):
            segment = script[i * segment_length: (i + 1) * segment_length]
            segment_features = features[i * segment_length: (i + 1) * segment_length]
            segment_embeddings = self.encoder.encode(segment, convert_to_tensor=True, device=device)
            input = torch.cat([segment_embeddings, segment_features], dim=1)
            output, (hidden, cell) = self.lstm(input, (hidden, cell))
            logits = self.classifier(output)
            _pred = logits.argmax(dim=1)
            pred.extend(_pred.cpu().tolist())
        
        return pred