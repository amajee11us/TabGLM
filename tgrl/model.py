import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from .archs.graph_models import GraphNetwork
from .archs.text_models import TextNetwork

class TGRLModel(nn.Module):
    def __init__(self, 
                 text_model_name,
                 graph_input_dim,
                 adjacency_matrix,
                 num_classes=2,
                 embedding_dim=512,
                 is_supervised=True):
        super(TGRLModel, self).__init__()

        # fetch the graph model - parameters are trainable
        self.graph_encoder = GraphNetwork(graph_input_dim,
                                          embedding_dim,
                                          adjacency_matrix)

        # fetch the text model - parameters are frozen during training
        self.text_encoder = TextNetwork(text_model_name) 

        self.is_supervised = is_supervised

        self.projection = nn.Linear((64 + 256 + 256) * embedding_dim, 768)

        # create a classifier for supervised learning
        if self.is_supervised:
            if num_classes > 2:
                self.activation = nn.Softmax(dim=1)
            else:
                self.activation = nn.Sigmoid()


            # self.classifier = nn.Sequential(
            #     nn.Linear((64 + 256 + 256) * embedding_dim, 1024),
            #     nn.Linear(1024, num_classes)
            # )
            self.classifier = nn.Linear(768, num_classes)


    def forward(self, text_in, graph_in, label=None):
        # Generate the graph_embeddings and text embeddings
        graph_embeddings = self.graph_encoder(graph_in)
        graph_embeddings = self.projection(graph_embeddings)

        # Generate the text_embeddings and text embeddings
        text_embeddings = self.text_encoder(text_in)

        if self.is_supervised:
            logits = self.classifier(graph_embeddings)
            return graph_embeddings, text_embeddings, logits

        return graph_embeddings, text_embeddings

