#!/bin/bash

# Function to update nested YAML value using yq
update_yaml() {
    local file="configs/$1"
    local key=$2
    local new_value=$3

    # Check if file exists
    if [[ ! -f $file ]]; then
        echo "File not found!"
        exit 1
    fi

    if [[ $new_value =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        # Update the key with the new numeric value using yq
        yq -y --in-place ".$key = $new_value" "$file"
    else
        # Update the key with the new string value using yq
        yq -y --in-place ".$key = \"$new_value\"" "$file"
    fi

    echo "Updated $key to $new_value in $file"
}

# Check if correct number of arguments is provided
if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <GPU_IDs> <Batch_Size>"
    exit 1
fi

# Fetch comma separated number of GPUs as input
GPUS=$1

# Set GPU dependent parameter - Batch Size
# Since early stopping in enabled, the number of epochs need not be varied
BATCH_SIZE=$2

# Pass the text encoder as an argument to the model
TEXT_ENCODER=$3

# Declare an array of strings of dataset names
# Note - These names are how they appear in the configs and not on openML
datasets=("pc3" \
        #   "ova_breast" \
          "mfeat_fourier" \
          "coil2000" \
          "kr_vs_kp" \
        #   "isolet" \
          "texture")
        #   "har" \
        #   "dilbert")
        #   "fashion_mnist", "cifar_10")

# Loop through the array and print each string
for dataset in "${datasets[@]}"; do
    CONFIG_FILE_PATH="config_${dataset}.yml"
    update_yaml "$CONFIG_FILE_PATH" "fit_config.batch_size" "$BATCH_SIZE"
    update_yaml "$CONFIG_FILE_PATH" "fit_config.text_encoder" "$TEXT_ENCODER"
    update_yaml "$CONFIG_FILE_PATH" "fit_config.text_tokenizer" "$TEXT_ENCODER"

    echo "Running train and eval on $dataset dataset"
    CUDA_VISIBLE_DEVICES=$GPUS python run.py config_${dataset}.yml

    echo "Done."
done
