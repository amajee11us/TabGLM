import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"

from tgrl.engine import TGRLMultiModalModel
from tgrl.data import preprocess_and_load_data
from tgrl.utils import get_config, store_metrics_as_csv, set_seed
import argparse
import wandb


def main(config_paths):
    for config_path in config_paths:
        data_config, fit_config = get_config(config_path)
        random_states = data_config[
            "random_states"
        ]  # Assumes both data_config and fit_config have the same random_states
        wandb_project_name = fit_config.get("project_name", None)
        local_save_path = fit_config.get("metrics_save_path", "tgrl_output_metrics.csv")

        for seed in random_states:
            if wandb_project_name:
                wandb.init(project=wandb_project_name)
                wandb.config.fit_config = fit_config

            # Set the random state in data_config and fit_config
            data_config["random_state"] = seed
            fit_config["random_state"] = seed
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
            tabularTrans = TGRLMultiModalModel(fit_config)

            tabularTrans.fit(
                X_train_norm=X_train_norm,
                X_val_norm=X_val_norm,
                y_train_enc=y_train_enc,
                y_val_enc=y_val_enc,
                categorical_encoder=cat_encoder,
            )
            metrics = tabularTrans.evaluate(X_test_norm, y_test_enc)
            print(f"Finished processing {config_path} with random seed {seed}")

            print("Aggregating Metrics ...")
            metrics["dataset"] = data_config["dataset"]
            metrics["seed"] = seed
            # Best epoch can be treated as the convergence point for the model
            metrics["best_epoch"] = wandb.summary[
                "best_epoch"
            ]  # Store the best epoch for comparisons
            store_metrics_as_csv(metrics, local_save_path)
            print("Done")

            wandb.finish()


if __name__ == "__main__":
    # Default list of YAML filenames
    default_yaml_names = ["config.yml", "config2.yml"]

    # Create the parser
    parser = argparse.ArgumentParser(
        description="Process some YAML configuration files."
    )
    # Add an argument, making it optional with nargs='*'
    parser.add_argument(
        "yaml_names", type=str, nargs="*", help="A list of YAML file names (optional)"
    )
    # Parse the arguments
    args = parser.parse_args()

    # Use command line arguments if provided, otherwise use the default list
    yaml_names_to_process = args.yaml_names if args.yaml_names else default_yaml_names
    # Get the current directory (should be where your notebook is)
    current_path = os.getcwd()
    # Create a list of full paths
    full_paths = [
        os.path.join(current_path, "configs", yaml_name)
        for yaml_name in yaml_names_to_process
    ]

    # Pass the list of YAML filenames to main function
    main(full_paths)
