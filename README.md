# Answerable_Suggestion_Model

## Overview

The `Answerable_Suggestion_Model` repository contains implementations and evaluations of various BART-based multitask learning (MTL) models for Answerable Suggestions based on unanswerable questions based on plausible answers. This project aims to improve the generation of answerable questions and the classification of text. The models are tested with different hyperparameters and datasets to enhance performance.

## Tasks and Models

### Tasks Overview

- **Task 21 Baseline:** These serve as reference points for evaluating model performance. The score to beat is 2109.
- **Task 22:** 2 MTL models were made for this task, one with a single classifier, having a score of 2321, and one with 2 classifiers, with a score of 2581.

### Model Configurations

1. **Baseline Model**
   - **Epochs:** 3
   - **Learning Rate:** 5e-5
   - **Optimizer:** Adam
   - **Dropout and Loss Weights:** Default settings.

2. **MTL Models**
   - **Initial Attempts:** Models with issues due to excessive epochs or data inconsistencies.
   - **Balanced Dataset Model:** Includes a balanced dataset and improved hyperparameters.
     - **Dropout:** Lower dropout for better classification.
     - **Loss Weights:** Updated to [1, 0.2, 0.2] for improved performance.

## Instructions for Replication

### Preparation

1. **Copy Model Directory**
   - Create a copy of the model directory.
   - Clear the contents in the directories as needed.

2. **File Management**
   - Before each run, delete all files that aren't present in the current directories, but for retroreader and SGNet, install and put their model files in their respective directories: retroreader (cls_model, model4), and SG-Net. 

### Training

1. **Update Files**
   - Edit the `bart-running(train)` file as needed.
   - Ensure the `Bart-Eval` file is updated with consistent `MTLModel` class definitions.

2. **Run Training**
   - Execute the `Bart-running` file for training.
   - If training crashes, use the following code to load from a checkpoint:
     ```python
     from safetensors.torch import load_file
     import os
     output_dir = './results-mtl-run/checkpoint-x'

     # Load the configuration
     config = BartConfig.from_pretrained(output_dir)

     # Reconstruct the model
     model = MTLModel(config)
     model.bart.model.encoder.load_state_dict(load_file(os.path.join(output_dir, "encoder.safetensors")))
     model.bart.model.decoder.load_state_dict(load_file(os.path.join(output_dir, "decoder.safetensors")))
     model.bart.model.shared.load_state_dict(load_file(os.path.join(output_dir, "shared.safetensors")))
     model.bart.lm_head.load_state_dict(load_file(os.path.join(output_dir, "lm_head.safetensors")))
     model.classifier_1.load_state_dict(load_file(os.path.join(output_dir, "classifier_1.safetensors")))
     model.classifier_2.load_state_dict(load_file(os.path.join(output_dir, "classifier_2.safetensors")))
     # Load the tokenizer
     tokenizer = BartTokenizer.from_pretrained(output_dir)
     ```
   - Paste this code right before `model.train`.
   - Note for the single classifier, remove one of the 2 classifiers, and just replace the name with classifier instead of classifier_x 

### Evaluation

1. **Run Evaluation Files**
   - After training, run the `Bart-Eval` file.
   - In the `MRC Eval` folder, run `U-Qpot-dev` and `Retro`.
   - In the `SG-Net` directory, run `Tags`.
   - Run `BARTCMD` to process results over SGNET.
   - Run `combining` in the `MRC Eval` folder.
   - Execute majority voting eval scripts under the base directory.

2. **Majority Voting and Metrics**
   - After finishing with `combining`, go to `Evaluation` on the main folder of the model, and run it to get generation results.
   - Then in the `MRC Eval` folder, run `Metrics` to get the classification performance.

## Datasets
- **Training Set:**
- `all_unans_salience.csv`
- **Original Dataset:** Initially used all data from the `unanswgen` dataset, but found to be imbalanced.
- **Balanced Dataset:** Includes answerable cases and a balanced distribution:
  - **Types and Counts:**
    - I: 3000
    - E: 3000
    - N: 3000
    - X: 3221
    - A: 3000
    - #: 2255
    - Ans: 3000
  - **Total:** 20,476 examples

- **Development Set:** 
- `dev_sal.csv`
- Includes unanswerable questions from the `squad_dev2.0` dataset with salient sentences:
  - **Types and Counts:**
    - I: 655
    - E: 2394
    - N: 818
    - X: 401
    - A: 1184
    - #: 493
  - **Total:** 5,945 examples
