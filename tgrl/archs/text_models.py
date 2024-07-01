from transformers import TapasTokenizer, TapasModel, AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn as nn
import pandas as pd

class TextNetwork(nn.Module):
    def __init__(self, encoder_name):
        super(TextNetwork, self).__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model and tokenizer
        if encoder_name == "tapas":
            self.tokenizer = TapasTokenizer.from_pretrained('google/tapas-base')
            self.model = TapasModel.from_pretrained('google/tapas-base')
        elif encoder_name == "tapex":
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/tapex-large-finetuned-tabfact")
            self.model = AutoModelForSequenceClassification.from_pretrained("microsoft/tapex-large-finetuned-tabfact")
    
    def forward(self, text_inputs, label=None):
        if label is not None:
            raise NotImplementedError("Supervised training on language model is not supported.")
        
        # Perform inference to get hidden states
        with torch.no_grad():
            outputs = self.model(**text_inputs)
            hidden_states = outputs.last_hidden_state
        
        # Aggregate the embeddings (e.g., by taking the mean across the token dimension)
        row_embeddings = hidden_states.mean(dim=1)  # Mean of hidden states
        
        return row_embeddings