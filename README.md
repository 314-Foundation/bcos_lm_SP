# SP in LLM


## Formatowanie i linting kodu Python w VS Code

W projekcie używamy **VS Code** z formatowaniem kodu Python przez **Black** oraz porządkowaniem importów przez **isort**. Dzięki temu kod jest automatycznie formatowany przy zapisie pliku, a importy są układane zgodnie z profilem kompatybilnym z Black.

Poniższą konfigurację należy wkleić do pliku ustawień VS Code:

```text
.vscode/settings.json
```

Jeśli plik lub katalog `.vscode` jeszcze nie istnieje, należy je utworzyć w głównym katalogu projektu.

```json
{
  "python.linting.pylintUseMinimalCheckers": false,
  "python.linting.banditPath": "",
  "python.linting.pylintEnabled": false,
  "[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.organizeImports": "explicit"
    },
    "editor.formatOnType": true
  },
  "isort.args": [
    "--profile",
    "black"
  ]
}
```

Po zapisaniu ustawień VS Code będzie automatycznie formatować pliki `.py` przy zapisie oraz uruchamiać organizowanie importów zgodnie z konfiguracją isort.


# README from B-cos LM

## Overview

B-cos LM is a modification of pre-trained language models to enhance interpretability while maintaining performance. Our implementation provides:

- **B-cos versions of BERT, DistilBERT, RoBERTa, GPT-2 and Llama models**  
- **Support for training B-cos and conventional models**  
- **Evaluation of B-cos and various post-hoc explanation methods**  

The core implementations are in:
- `bcos_lm/models/` – Contains B-cos model architectures  
- `bcos_lm/modules/` – Contains essential components for B-cos adaptation  

B-cos adaptations in the code are marked with `## bcos` for clarity.

## Environment

Our codes require `transformers==4.45.2`

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

## Decoder-only Models

Decoder-only model experiments (GPT-2 & Llama) can be run by executing `decoder_only_model_experiments.sh` in `decoder_experiments`.