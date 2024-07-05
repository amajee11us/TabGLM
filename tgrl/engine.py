import torch
import torch.optim as optim
import torch.backends.cudnn
import torch.cuda
from torch.optim import lr_scheduler

import os
import warnings
import wandb
import time
import numpy as np

warnings.simplefilter("ignore")

from tgrl.data import TGRLDataloader, create_dataloader
from tgrl.utils import get_config, find_repo_root, set_seed
from tgrl.losses import get_criterion, get_consistency_criterion
from tgrl.hooks import train_one_epoch, evaluate_model
from tgrl.models.multi_modal import TGRLModel


class TGRLMultiModalModel:
    def __init__(self, fit_config=None):

        if fit_config is None:
            # Navigate up to the parent directory (your_project_root)
            repo_root = find_repo_root()
            # Now, construct the path to the configuration file
            config_path = os.path.join(repo_root, "configs", "config.yml")
            _, fit_config = get_config(config_path)

        # wandb_project_name = fit_config.get('project_name', None)
        self.fit_config = fit_config
        # Now parse the config to initialize all the parameters
        self._parse_config(self.fit_config)

        # Initialize the pretrained model to NULL
        self.pretrained_model = None

        # set the random state 
        set_seed(self.random_state)

    def _parse_config(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.multimodal = config.get("multimodal", False)
        self.batch_size = config.get("batch_size", 224)

        best_model_path = config.get("best_model_path", "models/best_model.pth")

        # Navigate up to the parent directory (your_project_root)
        parent_directory = find_repo_root()

        # Now, construct the path to the configuration file
        self.best_model_path = os.path.join(parent_directory, best_model_path)

        # Hyper-parameters
        self.num_epochs = config.get("num_epochs", 10)
        self.learning_rate = config.get("learning_rate", 0.00001)
        self.random_state = config.get("random_state", 108)
        self.lr_patience = config.get("lr_patience", 10)
        self.alpha = config.get("alpha", 0.5)
        self.scheduler = config.get("scheduler", "plateau")
        self.early_stopping_patience = config.get("early_stopping_patience", 15)

        # Multi-modal pipeline hooks
        self.text_encoder = config.get("text_encoder", "tapas")
        self.text_tokenizer = config.get("text_tokenizer", "tapas")

        # Logistic Parameters
        self.verbose = config.get("verbose", True)
        self.cache_dir = config.get("cache_dir", "/scratch/acf15929tu/tgrl")

        # Representation specific parameters
        self.consistency = config.get("consistency", True)
        self.consistency_loss_type = config.get("consistency_loss_type", "mse")

        # Globally used tags
        self.task_type = self.fit_config.get("task_type", None)

    def _data_transformation(self, X_train_norm, X_val_norm, y_train_enc, y_val_enc):

        print("Processing Train Dataset ...")
        self.train_dataset = TGRLDataloader(
            X_train_norm, y_train_enc, multimodal=True, text_encoder="tapas"
        )
        print("Done.")

        print("Processing Val Dataset ...")
        self.val_dataset = TGRLDataloader(
            X_val_norm, y_val_enc, multimodal=True, text_encoder="tapas"
        )
        print("Done.")

        print("Create dataloaders...")
        self.trainloader = create_dataloader(
            self.train_dataset, "train", self.batch_size
        )
        self.valloader = create_dataloader(self.val_dataset, "val", self.batch_size)
        print("Done.")

    def _model_factory(self, y_train_enc):
        # Fetch the losses - Note we have two types of losses
        self.criterion = get_criterion(self.task_type, y_train_enc).to(self.device)
        if self.consistency:
            self.consistency_criterion = get_consistency_criterion(
                self.consistency_loss_type
            ).to(self.device)
        else:
            self.consistency_criterion = None  # We need this to add checks later

        adj_matrix = torch.FloatTensor(self.train_dataset.adjacency_matrix).to(
            self.device
        )
        # Fetch model - The TGRL multi-modal model
        self.model = TGRLModel(
            text_model_name=self.text_encoder,
            graph_input_dim=1,
            adjacency_matrix=adj_matrix,
            num_classes=self.num_classes,
            embedding_dim=adj_matrix.shape[0],
            is_supervised=True,
        )

        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs!")
            self.model = torch.nn.DataParallel(self.model)
            self.model = self.model.cuda()

        # Fetch Optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)

    def _train(self, verbose=True):
        if self.scheduler == "plateau":
            scheduler = lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=self.lr_patience,
                verbose=verbose,
            )
        else:
            scheduler = None  # Replace with the actual scheduler if needed

        best_loss = float("inf")
        best_epoch = 0
        early_stopping_counter = 0

        for epoch in range(self.num_epochs):
            start_time = time.time()  # Start time of the epoch
            train_one_epoch(
                self.model,
                self.trainloader,
                self.criterion,
                self.consistency_criterion,
                self.optimizer,
                self.task_type,
                epoch,
                self.device,
                consistency=self.consistency,
                is_supervised=self.multimodal,
                alpha=self.alpha,
            )

            val_loss, metrics = evaluate_model(
                self.model,
                self.valloader,
                self.criterion,
                self.task_type,
                self.device,
                "val",
                epoch=epoch,
                scheduler=scheduler,
                is_supervised=self.multimodal,
            )

            # Check for improvement
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                early_stopping_counter = 0
                torch.save(self.model.state_dict(), self.best_model_path)
                if verbose:
                    print(f"Epoch {epoch}: Validation improved. Loss: {best_loss:.2f}.")
            else:
                early_stopping_counter += 1
                if verbose:
                    print(
                        f"Epoch {epoch}: No improvement. Early stopping counter: {early_stopping_counter}/{self.early_stopping_patience}."
                    )

            if early_stopping_counter >= self.early_stopping_patience:
                if verbose:
                    print(f"Early stopping triggered at epoch {epoch}.")
                break

            end_time = time.time()  # End time of the epoch
            epoch_duration = end_time - start_time

            print(f"Epoch {epoch} completed in {epoch_duration:.2f} seconds")

        wandb.summary["best_epoch"] = best_epoch
        if verbose:
            print(f"Training completed. Best epoch {best_epoch:.2f}.")

    def fit(self, X_train_norm, X_val_norm, y_train_enc, y_val_enc, transform=True):
        # Calculate the Number of classes
        self.num_classes = (
            len(np.unique(y_train_enc)) if self.task_type == "multi_class" else 1
        )

        print("Transforming data...")
        self._data_transformation(X_train_norm, X_val_norm, y_train_enc, y_val_enc)

        print("Loading the model ...")
        self._model_factory(y_train_enc)

        print("Starting training...")
        self._train(verbose=self.verbose)

    def predict(self, X):
        self.model.eval()
        predictions = []

        if isinstance(X, torch.utils.data.DataLoader):  # If input is a DataLoader
            with torch.no_grad():
                for batch in X:
                    batch_predictions = self.model(batch[0].to(self.device))
                    predictions.append(batch_predictions.cpu())
            # Concatenate all batch predictions
            predictions = torch.cat(predictions).numpy()
        else:  # If input is a dataframe or tensor
            with torch.no_grad():
                predictions = self.model(X.to(self.device))
            predictions = predictions.cpu().numpy()

        return predictions

    def evaluate(self, X, y, shuffle_channels_flag=False):
        model = self.model
        model.load_state_dict(torch.load(self.best_model_path))

        dataset = TGRLDataloader(X, y, multimodal=True, text_encoder="tapas")
        testloader = create_dataloader(self.val_dataset, "test", self.batch_size)

        loss, metrics = evaluate_model(
            model,
            testloader,
            self.criterion,
            self.task_type,
            self.device,
            "test",
            is_supervised=True,
        )
        if self.task_type == "regression":
            wandb.summary[f"final R2 {set[i][1]}"] = metrics["test_r2"]
        else:
            wandb.summary["final accuracy"] = metrics["test_accuracy"]
            wandb.summary["final f1"] = metrics["test_f1"]
            wandb.summary["final auroc"] = metrics["test_auroc"]

        # Placeholder for local collection and storage of results
        return metrics
