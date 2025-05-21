# Project README

## Introduction

This project focuses on federated learning utilizing the BERT model. It is designed to support various natural language processing tasks, including those from the GLUE benchmark and the SQuAD dataset.

## Workflow

This section outlines the project's structure and how to run the training scripts.

### Training Scripts

The project includes several Python scripts for training models:

*   `glue_teachermodel_train.py`: Trains a teacher model on the GLUE dataset.
*   `squad_teachermodel_train.py`: Trains a teacher model on the SQuAD dataset.
*   `glue_fedmodel_train.py`: Performs federated training of a student model on the GLUE dataset using knowledge distillation from a teacher model.
*   `squad_fedmodel_train.py`: Performs federated training of a student model on the SQuAD dataset using knowledge distillation from a teacher model.
*   `glue_fedavg_baseline_train.py`: Performs federated training on the GLUE dataset using the FedAvg algorithm (baseline).
*   `squad_fedavg_baseline-.py`: Performs federated training on the SQuAD dataset using the FedAvg algorithm (baseline).

### Core Directories

*   `modeling/`: This directory contains custom BERT model implementations, including variations tailored for techniques like CoFi pruning.
*   `pipeline/`: This directory houses the infrastructure for the training process. It includes:
    *   Training argument definitions (e.g., learning rate, batch size).
    *   Entry point scripts that orchestrate the training for different tasks and models.
    *   Trainer classes that manage the model training and evaluation loops.

### Basic Usage

To run any of the training scripts, use the Python interpreter. For example, to train a teacher model on GLUE:

```bash
python glue_teachermodel_train.py [arguments]
```

Similarly, for federated training on SQuAD with FedAvg:

```bash
python squad_fedavg_baseline-.py [arguments]
```

Replace `[arguments]` with the desired command-line arguments specific to each script (e.g., for setting hyperparameters, data paths, etc.).
