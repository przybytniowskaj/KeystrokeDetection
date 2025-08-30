# KeystrokeDetection

## Overview

In the pursuit of stronger digital security, research has largely focused on encryption, biometrics, and hardened systems. Yet devices still emit subtle traces, side channels, that attackers can exploit, with keystroke sounds posing a particularly concerning risk.

This thesis evaluates a range of neural network architectures for acoustic keystroke recognition across different environments, background conditions, and key sets. The models achieved 92\% accuracy on publicly available recordings and set a new state-of-the-art performance on existing benchmarks.

Beyond accuracy, the work investigates how training data, model scale, and noise exposure affect generalization, and explores whether large language models can use semantics to repair errors in raw predictions. The study highlights both the possibilities and limits of acoustic side-channel attacks, pointing to effective defense strategies such as noise injection, hardware modifications, or keyboard design.

---

This repository contains all the necessary scripts, notebooks, and resources for conducting experiments on keystroke detection. It includes data processing, model training, evaluation, and analysis. Below, you'll find a detailed explanation of the folder structure, how to run experiments, and the tools provided in this repository.

---

## Folder Structure

The repository is organized as follows:

```
KeystrokeDetection/
├── data/                           # Folder for storing datasets
│   ├── final/                      # Preprocessed data files
├── notebooks/                      # Jupyter notebooks for data processing, evaluation, and analysis
│   ├── data_processing.ipynb       # Notebook for data preprocessing
│   ├── evaluation.ipynb            # Notebook for evaluation and metrics
│   ├── model_architectures.ipynb   # Notebook for testing different model architectures
├── src/                            # Python scripts for training and evaluation
│   ├── constants/                  # Folder for all static variables
│   │   ├── evaluation.py           # LLM evaluation constants
│   │   ├── loading.py              # Data loading constants
│   │   └── segmentation.py         # Audio segmentation constants
│   ├── models/                     # Folder for architecture definition
│   │   ├── moat.py                 # MOAT definition
│   │   ├── coatnet.py              # CoAtNet definition
│   │   └── swin_transformer.py     # Swin Transformer definition
│   ├── utils/                      # Folder with all function definitions
│   │   ├── evaluation.py           # Script with functions for evaluation
│   │   ├── loading.py              # Script with functions for data loading
│   │   ├── segmentation.py         # Script with functions for audio segmentation
│   │   └── train_eval.py           # Script with functions for training loop
│   └── train_model.py              # Script for training model
├── configs/                        # Configuration files for experiments
│   ├── config.yaml                 # Example configuration file
|   └── return_unique_param_set.py  # Script to define parameters when training arrays
├── run_experiment.sh               # Bash script for model training
├── .env                            # Environment file for storing API keys
├── requirements.txt                # Python dependencies
├── requirements_mac.txt            # Python dependencies for MacOS
├── README.md                       # Repository documentation
└── LICENSE                         # License for the project
```

---

## How to Run Experiments

### 1. **Setup Environment**
   - Clone the repository:
     ```bash
     git clone https://github.com/your-username/KeystrokeDetection.git
     cd KeystrokeDetection
     ```
   - Install dependencies (depending on your OS):
     ```bash
     pip install -r requirements.txt
     ```
     or
     ```bash
     pip install -r requirements_mac.txt
     ```
   - Create a .env file in the root directory and add your Gemini and Cloudfare API keys

### 2. **Data Processing**
   - Use the `notebooks/data_processing.ipynb` notebook to preprocess the raw data. This notebook includes steps for cleaning, feature extraction, and preparing the data for training.

### 3. **Training**
   - Modify the `configs/config.yaml` file to set up your experiment parameters (e.g., dataset path, model architecture, hyperparameters).
   - Run the training script using below:
     ```bash
     python src/train_model.py --config config.yaml
     ```

---

## Notebooks

The `notebooks/` folder contains Jupyter notebooks for various stages of the pipeline:

1. **`data_processing.ipynb`**:
   - Handles data cleaning, feature extraction, and preparation for training.

2. **`evaluation.ipynb`**:
   - Includes scripts for evaluating model performance, reading from WanDb experiments, and performing text reconstruction with LLM

3. **`model_architectures.ipynb`**:
   - Explores different model architectures and their size.


You can modify these scripts to customize the commands or add additional parameters.

---
## Paper
This work was described in [Paper](https://github.com/przybytniowskaj/KeystrokeDetectionDocument/blob/main/thesis-en.pdf).

---

## License

This repository is licensed under the MIT License. You are free to use, modify, and distribute the code as long as you include the original license.

---

Feel free to reach out if you have any questions or issues!