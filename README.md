# TGRL: Tabular Graph-Text Representation Learning with Consistency Minimization
Tabular Graph-Text Representation Learning with Consistency Minimization. 
This method demonstrates a multi-modal model which aims to improve classification/ regression performance on tabular datasets by encoding both structural and semantic features in tabular datasets.

## Installation
```bash
conda create --name tgrl python=3.12.4
conda activate tgrl
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=11.8 -c pytorch -c nvidia

sudo apt-get install jq

git clone <REPO_URL>.git
cd TGRL

pip install -e .
```

**Note**: We have tested our code for CUDA toolkit 11.8, 12.1 and 12.4. 

## Configuration

All configurations are located in the configs/config.yml file. Before running the model, ensure that the configurations are set to your preference including dataset details and training parameters.

## Experimentation Setup

To perform training and evaluation follow the following steps.

```bash
CUDA_VISIBLE_DEVICES=<GPU_IDs> python run.py <CONFG_FILE>
```

```GPU_IDs``` - Comma separated integers indicating the GPUs to train on. TGRL supports multi-GPU training.

```CONFIG_FILE``` - Config file demonstrating data, training and evaluation parameters for each dataset.

To Execute all experiments (subject to resource availability) please use the below command.

```
bash run_all_experiments.sh <GPU_IDs> <BATCH_SIZE>
```

There are two key command line arguments -

```GPU_IDs``` represent comma separated list of GPUs.

```BATCH_SIZE``` represents the number of records in a batch (vary based on GPU capacity).
