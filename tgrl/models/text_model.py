import torch
import torch.nn as nn
import pandas as pd

from tgrl.models import text_model_dict as model_dict


class TextNetwork(nn.Module):
    def __init__(self, encoder_name):
        super(TextNetwork, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model and tokenizer
        if encoder_name not in list(model_dict.keys()):
            raise Exception(
                "Text Model of type : {}, does not exist.".format(encoder_name)
            )

        # TODO : For TAPEX just use the encoder and do not load the parameters of decoder
        self.encoder_type = encoder_name
        self.model = model_dict[encoder_name]["model"]
        self.tokenizer = model_dict[encoder_name]["tokenizer"]
        self.embedding_dim = model_dict[encoder_name]["embedding_dim"]

        self.__init__parameters()
        print(self.model)

    def __init__parameters(self):
        if self.encoder_type == "tapex":  # TAPEX has both encoder and decoder
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            for param in self.model.encoder.parameters():
                param.requires_grad = False

    def forward(self, text_inputs, label=None):
        if label is not None:
            raise NotImplementedError(
                "Supervised training on language model is not supported."
            )

        # Perform inference to get hidden states
        with torch.no_grad():
            outputs = self.model(**text_inputs)
            if self.encoder_type == "tapex":  # TAPEX has both encoder and decoder
                hidden_states = outputs.encoder_last_hidden_state
            else:
                hidden_states = outputs.last_hidden_state

        # Aggregate the embeddings (e.g., by taking the mean across the token dimension)
        row_embeddings = hidden_states.mean(dim=1)  # Mean of hidden states

        return row_embeddings
