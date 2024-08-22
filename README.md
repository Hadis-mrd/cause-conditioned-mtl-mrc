# Answerable_Suggestion_Model

## Overview

The `Answerable_Suggestion_Model` repository contains implementations and evaluations of various BART-based multitask learning (MTL) models for AthenaGuard. This project aims to improve the generation of answerable questions and the classification of text. The models are tested with different hyperparameters and datasets to enhance performance.

## Tasks and Models

### Tasks Overview

- **Task 20 & Task 21 Baselines:** These serve as reference points for evaluating model performance. The score to beat is 2103.
- **Task 22:** Builds upon previous models with various configurations to enhance results. Focuses on dropout rates, loss weights, and dataset balancing.

### Model Configurations

1. **Baseline Model**
   - **Epochs:** 3
   - **Learning Rate:** 5e-5
   - **Optimizer:** Adam
   - **Dropout and Loss Weights:** Default settings.

2. **Updated Models**
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
   - In the `MRC Eval` directory, delete specified files.
   - In the `sg-net` directory, delete necessary files.

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

### Evaluation

1. **Run Evaluation Files**
   - After training, run the `Bart-Eval` file.
   - In the `MRC Eval` folder, run `U-Qpot-dev` and `Retro`.
   - In the `SG-Net` directory, run `Tags`.
   - Run `BARTCMD` to process results over SGNET.
   - Run `combining` in the `MRC Eval` folder.
   - Execute majority voting eval scripts under the base directory.

## Training Data

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

- **Development Set:** Includes unanswerable questions from the `squad_dev2.0` dataset with salient sentences:
  - **Types and Counts:**
    - I: 655
    - E: 2394
    - N: 818
    - X: 401
    - A: 1184
    - #: 493
  - **Total:** 5,945 examples

## Results and Findings

- **Baseline Results:** Established initial performance benchmarks.
- **Single Classifier Model:** Evaluated with 6 epochs.
- **Two Classifier Model:** Compared performance with and without the additional classifier.

**Interpretation:** 
- Observed overfitting with no information, leading to better question generation when the dataset was diverse. This improved the generator's focus on generating answerable questions and salient sentences.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
