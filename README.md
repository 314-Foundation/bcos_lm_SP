# B-cos LM

This repository contains the implementation of **B-cos LM**, as introduced in the paper **"B-cos LM: Efficiently Transforming Pre-trained Language Models for Improved Explainability"**.

## Overview

B-cos LM is a modification of pre-trained language models to enhance interpretability while maintaining performance. Our implementation provides:

- **B-cos versions of BERT, DistilBERT, and RoBERTa**  
- **Support for training B-cos and conventional models**  
- **Evaluation of B-cos and various post-hoc explanation methods**  

The core implementations are in:
- `bcos_lm/models/` – Contains B-cos model architectures  
- `bcos_lm/modules/` – Contains essential components for B-cos adaptation  

B-cos adaptations in the code are marked with `## bcos` for clarity.

## Getting Started

### 1. Training B-cos LM

To train a B-cos LM model, run:

```bash
bash train_bcos_models.sh
```

You can specify:
- **Model** (e.g., BERT, DistilBERT, RoBERTa)  
- **Dataset**  
- **Hyperparameters**  

Modify `train_bcos_models.sh` to customize these settings.

### 2. Generating Explanations

To generate explanations using B-cos and other explanation methods, run:

```bash
bash generate_explanations.sh
```

You can specify explanation methods to use.

### 3. Perturbation-based Evaluation

To evaluate the model using perturbation-based methods, run:

```bash
bash run_perturbation_evaluation.sh
```

### 4. Sequence Pointing Game (SeqPG) Evaluation

1. **Generate SeqPG examples using conventional models:**
   ```bash
   bash create_pointing_game_examples.sh
   ```
2. **Evaluate using SeqPG:**
   ```bash
   bash run_pointing_game_evaluation.sh
   ```

