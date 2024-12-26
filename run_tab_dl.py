import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

from tabglm.data.utils import preprocess_and_load_data
from tabglm.utils import get_config, store_metrics_as_csv, set_seed, log_metrics
import argparse
import wandb

import pandas as pd
from pytorch_tabular import TabularModel
from pytorch_tabular.models import TabTransformerConfig, FTTransformerConfig, NodeConfig
from pytorch_tabular.config import (
    DataConfig,
    OptimizerConfig,
    TrainerConfig,
)
from pytorch_tabular.utils import get_class_weighted_cross_entropy

from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error, roc_auc_score

def encode_df(df, numerical_columns, categorical_columns, categorical_encoder):
    raw_data_numerical = df[numerical_columns]
    
    encoded_raw_data_categorical = categorical_encoder.transform(df[categorical_columns])
    
    encoded_raw_data_categorical_df = pd.DataFrame(
        encoded_raw_data_categorical,
        columns=categorical_encoder.get_feature_names_out(categorical_columns),
    )
    
    df = pd.concat([raw_data_numerical, encoded_raw_data_categorical_df], axis=1)
    
    return df

def main(config_paths, model_name):
    for config_path in config_paths: 
        data_config, fit_config = get_config(config_path)
        random_states           = data_config['random_states']  # Assumes both data_config and fit_config have the same random_states
        wandb_project_name      = fit_config.get('project_name', None)
        wandb_project_name      = "TabGLM_{}".format(model_name)
        local_save_path         = fit_config.get('metrics_save_path', '{}_output_metrics.csv'.format(model_name))

        print(data_config['task_type'])
        if data_config['task_type']== "regression": 
            task_type= "regression"
        else: 
            task_type= "classification"
            
        # Select Experiment Settings
        if model_name == "Tab-transformer":
            TabModelConfig = TabTransformerConfig
            max_epochs = 500
        elif model_name == "FT-transformer":
            TabModelConfig = FTTransformerConfig
            max_epochs = 240
        elif model_name == "NODE":
            TabModelConfig = NodeConfig
            max_epochs = 500
        else:
            raise Exception("Model Config Not found.")

        print(random_states)
        for seed in random_states:
            print(seed)
            # Set the random state in data_config and fit_config
            data_config['random_state'] = seed
            fit_config['random_state'] = seed
            (
                X_train_norm,
                X_val_norm,
                X_test_norm,
                y_train_enc,
                y_val_enc,
                y_test_enc,
                scaler_y,
                cat_encoder,
            ) = preprocess_and_load_data(data_config)
            
            # Extract Numerical columns and separate
            numerical_columns = X_train_norm.select_dtypes(include=["number"]).columns.tolist()
            
            # Extract Categorical columns and Encode
            categorical_columns = X_train_norm.select_dtypes(include=["object", "category"]).columns.tolist()

            if cat_encoder: 
                X_train_norm = encode_df(X_train_norm, numerical_columns, categorical_columns, cat_encoder)
                X_val_norm = encode_df(X_val_norm, numerical_columns, categorical_columns, cat_encoder)
                X_test_norm = encode_df(X_test_norm, numerical_columns, categorical_columns, cat_encoder)

            def clean_feature_name(name):
                # Replace invalid characters with an underscore or remove them
                return str(name).replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_')
            
            def clean_feature_names(feature_names):
                # Apply the cleaning function to all feature names
                return [clean_feature_name(name) for name in feature_names]

            # Clean feature names
            cleaned_feature_names_train = clean_feature_names(X_train_norm.columns)
            cleaned_feature_names_val = clean_feature_names(X_val_norm.columns)
            cleaned_feature_names_test = clean_feature_names(X_test_norm.columns)
            
            # Assign cleaned feature names back to the DataFrames
            X_train_norm.columns = cleaned_feature_names_train
            X_val_norm.columns = cleaned_feature_names_val
            X_test_norm.columns = cleaned_feature_names_test

            # Convert numpy arrays to DataFrames
            X_train_df = pd.DataFrame(X_train_norm)
            X_val_df = pd.DataFrame(X_val_norm)
            X_test_df = pd.DataFrame(X_test_norm)            

            # PyTorch Tabular configuration
            data_config_pt = DataConfig(
                target=[data_config['target']],
                continuous_cols=[col for col in X_train_df.columns],
                categorical_cols=[]
            )
            
            trainer_config = TrainerConfig(
                auto_lr_find=True,  # Runs the LRFinder to automatically derive a learning rate
                batch_size=256,
                max_epochs=500,
                accelerator='gpu',  # Use 'gpu' for GPU training
                devices=1,  # Automatically choose GPU if available
                seed = seed
            )

            optimizer_config = OptimizerConfig()

            model_config = TabModelConfig(
                task=task_type,
                seed = seed
                #layers="1024-512-512",  # Number of nodes in each layer
                #activation="LeakyReLU",  # Activation between each layers
                #learning_rate=1e-3,
            )

            tabular_model = TabularModel(
                data_config=data_config_pt,
                model_config=model_config,
                optimizer_config=optimizer_config,
                trainer_config=trainer_config
            )
            
            # Prepare training data
            train_data = pd.concat([X_train_df, pd.Series(y_train_enc, name=data_config['target'])], axis=1)
            val_data = pd.concat([X_val_df, pd.Series(y_val_enc, name=data_config['target'])], axis=1)
            
            wandb.init(project=wandb_project_name)
            wandb.config.random_state = seed

            # Log model name and parameters
            wandb.config.model_name = model_name
            wandb.config.dataset = data_config["dataset"]

            if data_config['task_type']== "multi_class": 
                weighted_loss = get_class_weighted_cross_entropy(train_data[data_config['target']].values.ravel(), mu=0.1)
                # Train the model
                tabular_model.fit(train=train_data, validation=val_data, loss = weighted_loss)
            else: 
                tabular_model.fit(train=train_data, validation=val_data)

            # Predict outputs
            y_test_pred = tabular_model.predict(X_test_df)['c_prediction']

            if data_config["task_type"] == "regression":
                y_test_proba = None
            elif data_config["task_type"] == "binary":
                # Predict outputs
                y_test_proba = tabular_model.predict(X_test_df)['c_1_probability']
            elif data_config["task_type"] == "multi_class":
                # Predict outputs
                y_test_proba = tabular_model.predict(X_test_df).iloc[:, :-1]
                            
            print(f"Finished processing {config_path} with random seed {seed}")
            print(data_config['task_type'])

            print("Aggregating Metrics ...")
            metrics = log_metrics(data_config['task_type'], y_test_enc, y_test_pred, y_test_proba, loss=0.0, phase="test")

            print("Done")
            
            print("Done")
            metrics["dataset"] = data_config["dataset"]
            metrics["model_Name"] = model_name
            metrics["seed"] = seed
            store_metrics_as_csv(metrics, local_save_path) 

            print("Done")
            wandb.finish()


if __name__ == "__main__":
    # Default list of YAML filenames
    default_yaml_names = ["config_isolet.yml", "config_pc3.yml"]
    
    # Create the parser
    parser = argparse.ArgumentParser(description='Process some YAML configuration files.')
    # Add an argument, making it optional with nargs='*'
    parser.add_argument('yaml_names', type=str, nargs='*', help='A list of YAML file names (optional)')
    # Add an argument, to select model 
    parser.add_argument('model_name', type=str, choices=['Tab-transformer', 'FT-transformer', 'NODE'], help='Model Name')

    # Parse the arguments
    args = parser.parse_args()
    
    # Use command line arguments if provided, otherwise use the default list
    yaml_names_to_process = args.yaml_names if args.yaml_names else default_yaml_names
    # Get the current directory (should be where your notebook is)
    current_path = os.getcwd()
    # Create a list of full paths
    full_paths = [os.path.join(current_path, 'configs', yaml_name) for yaml_name in yaml_names_to_process]

    # Pass the list of YAML filenames to main function
    main(full_paths, args.model_name)
