import torch.nn as nn
import torch

from tabglm.models.graph_model import GraphNetwork
from tabglm.models.text_model import TextNetwork


class TabGLMModel(nn.Module):
    def __init__(
        self,
        text_model_name,
        graph_input_dim,
        adjacency_matrix,
        num_classes=2,
        embedding_dim=512,
        is_supervised=True,
        text_only=False,
    ):
        super(TabGLMModel, self).__init__()

        self.text_only_network = text_only

        if not self.text_only_network:
            # fetch the graph model - parameters are trainable
            self.graph_encoder = GraphNetwork(
                graph_input_dim, embedding_dim, adjacency_matrix
            )

        # fetch the text model - parameters are frozen during training
        self.text_encoder = TextNetwork(text_model_name)
        self.text_encoder_output_dim = self.text_encoder.embedding_dim

        self.is_supervised = is_supervised

        if not self.text_only_network:
            self.graph_projection = nn.Linear(
                (64 + 256 + 256) * embedding_dim, self.text_encoder_output_dim
            )
        
        self.text_projection = nn.Linear(
            self.text_encoder_output_dim, self.text_encoder_output_dim
        )

        # create a classifier for supervised learning
        if self.is_supervised:
            if num_classes > 2:
                self.activation = nn.Softmax(dim=1)
            else:
                self.activation = nn.Sigmoid()

            self.classifier = nn.Linear(self.text_encoder_output_dim, num_classes)

    def forward(self, text_in, graph_in, label=None):
        graph_embeddings = None
        if not self.text_only_network:
            # Generate the graph_embeddings and text embeddings
            graph_embeddings = self.graph_encoder(graph_in)
            graph_embeddings = self.graph_projection(graph_embeddings)

        # Generate the text_embeddings and text embeddings
        text_embeddings = self.text_encoder(text_in)
        text_embeddings = self.text_projection(text_embeddings)

        if self.is_supervised:
            if self.text_only_network:
                logits = self.classifier(text_embeddings)
            else:
                logits = self.classifier(graph_embeddings)
            return graph_embeddings, text_embeddings, logits

        return graph_embeddings, text_embeddings
