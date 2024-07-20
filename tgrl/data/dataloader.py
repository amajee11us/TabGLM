import copy
import torch
import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from torch.utils.data import Dataset
import copy

from tgrl.models import text_model_dict

# Ignore the HF warnings
warnings.simplefilter("ignore")


class TGRLDataloader(Dataset):
    """
    Table-Graph Multi-Modal DataLoader
    Joint dataloader for the TGRL model.
    """

    def __init__(
        self,
        raw_data,
        labels,
        multimodal,
        text_encoder="tapas",
        categorical_encoder=None,
    ):

        self.raw_data = raw_data
        self.labels = labels

        # Convert categorical data into OneHotEncoded embeddings
        self.raw_data_graph = copy.deepcopy(self.raw_data)
        if categorical_encoder is not None:
            # Extract Numerical columns and separate
            numerical_columns = self.raw_data_graph.select_dtypes(
                include=["number"]
            ).columns.tolist()
            raw_data_numerical = self.raw_data_graph[numerical_columns]

            # Extract Categorical columns and Encode
            categorical_columns = self.raw_data_graph.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

            encoded_raw_data_categorical = categorical_encoder.transform(
                self.raw_data_graph[categorical_columns]
            )
            encoded_raw_data_categorical_df = pd.DataFrame(
                encoded_raw_data_categorical,
                columns=categorical_encoder.get_feature_names_out(categorical_columns),
            )

            self.raw_data_graph = pd.concat(
                [raw_data_numerical, encoded_raw_data_categorical_df], axis=1
            )

        self.adjacency_matrix, _, _ = self.compute_adjacency_matrix(
            self_loop_weight=20, threshold=0.2
        )
        self.graph_data = self.generate_graph_tensors()

        self.multimodal = multimodal  # required switch during inferencing

        # Get text tokenizer
        if text_encoder in list(text_model_dict.keys()):
            self.tokenizer = text_model_dict[text_encoder]["tokenizer"]
        else:
            raise Exception("Text tokenizer not found !")

        # Store the label tensor
        self.label_tensor = torch.FloatTensor(self.labels)

    def compute_adjacency_matrix(self, self_loop_weight, threshold):
        index_to_name = {i: n for i, n in enumerate(self.raw_data_graph.columns)}
        name_to_index = {n: i for i, n in enumerate(self.raw_data_graph.columns)}

        adj_matrix = np.zeros(
            (len(self.raw_data_graph.columns), len(self.raw_data_graph.columns)),
            dtype=float,
        )

        for col_1 in self.raw_data_graph.columns:
            for col_2 in self.raw_data_graph.columns:
                if col_1 == col_2:
                    adj_matrix[name_to_index[col_1], name_to_index[col_2]] = (
                        self_loop_weight
                    )
                else:
                    corr, _ = pearsonr(
                        self.raw_data_graph[col_1], self.raw_data_graph[col_2]
                    )
                    if np.isnan(corr) or abs(corr) < threshold:
                        adj_matrix[name_to_index[col_1], name_to_index[col_2]] = 0
                    else:
                        adj_matrix[name_to_index[col_1], name_to_index[col_2]] = corr

        return adj_matrix, index_to_name, name_to_index

    def generate_graph_tensors(self):
        data = copy.deepcopy(self.raw_data_graph)
        data_np = data.to_numpy()

        # Convert entire data and labels arrays to tensors
        data_tensor = torch.FloatTensor(data_np)

        # Reshape data tensor to match the required shape
        graph_tensor = data_tensor.view(len(data), self.adjacency_matrix.shape[0], 1)

        return graph_tensor

    def extract_sample_text_encoding(self, row):
        # Prepare the table (single row as a table)
        table_df = row.to_frame().T

        # Tokenize the input (no need for a question)
        inputs = self.tokenizer(
            table=table_df.reset_index(drop=True).astype(str),
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        inputs = {key: val.squeeze(0) for key, val in inputs.items()}

        return inputs

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        datapoint_idx = self.raw_data.iloc[idx]

        label = self.label_tensor[idx]

        graph_encoding = self.graph_data[idx]

        if not self.multimodal:
            return graph_encoding, label

        text_encoding = self.extract_sample_text_encoding(datapoint_idx)

        return graph_encoding, text_encoding, label
