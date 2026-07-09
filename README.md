# Semantic Pullbacks for Language Models

Internal working repository for transferring **Semantic Pullbacks (SP)** from vision models to **encoder-only language model classifiers**.

The main goal is to combine ideas and code from:

* Semantic Pullbacks: https://github.com/314-Foundation/SemanticPullbacks
* B-cos LM: https://github.com/Ewanwong/bcos_lm

The planned target is a lightweight, refactored repository for experimenting with **Semantic Pullbacks for text classification**, inspired by the setup of B-cos LM.

## Project context

Semantic Pullbacks have recently shown strong results for explaining vision models. They provide faithful, stable and class-conditional explanations, as well as useful counterfactual directions, without modifying the forward model.

This project asks whether the same idea can be transferred to language model classification.

In the language setting, the input is no longer an image but a sequence of token embeddings:

```text
x = (x_1, ..., x_n)
```

This means that explanations first live in embedding space and then need to be aggregated into token-level attributions.

We focus on encoder-only classification models and compare Semantic Pullbacks against gradient-based and B-cos baselines.

## Main research questions

We want to evaluate whether the favourable empirical properties of Semantic Pullbacks observed in vision survive the transfer to text:

* faithfulness,
* stability,
* target-specificity,
* computational efficiency,
* usefulness of token-level explanations.

Counterfactuals are currently optional and may be left out of scope. Text counterfactual generation is non-trivial because nearest-neighbour moves in embedding space do not directly guarantee fluency, phrase consistency or correct handling of positional embeddings. We may still run exploratory experiments, but a dedicated follow-up project may be more appropriate.

## Related work and resources

### Core papers

* **Semantic Pullbacks:** https://arxiv.org/abs/2507.22832
* **B-cos LM:** https://arxiv.org/abs/2502.12992
* B-cos: https://www.semanticscholar.org/reader/cb7738a3b0a7df34d4febee9295b08d835f98e10
* B-cosification: https://proceedings.neurips.cc/paper_files/paper/2024/file/72d50a87b218d84c175d16f4557f7e12-Paper-Conference.pdf

### Project notes

* Overleaf project: https://www.overleaf.com/project/6a43f43b543bbdbc64995e6d
* Soft introduction to Pullbacks: https://cogita.ai/how-deep-neural-networks-see-the-world/

## Implementation plan

The first milestone is to merge the relevant parts of the Semantic Pullbacks repository with the B-cos LM codebase.

In particular, we need to implement Semantic Pullbacks for language models.

There are two possible implementation directions:

1. Follow the original SP repository design:

   * use a `surrogates.py` file,
   * define a `SURROGATE_CLASS_MAP`,
   * replace selected modules with surrogate modules that implement a softened backward pass.

2. Reuse and adapt ideas from B-cos LM:

   * B-cos LM already performs a related kind of module modification through dynamic multiplication,
   * however, it does not currently use temperature-based softened backward passes,
   * it may be cleaner to design a unified abstraction instead of directly copying the SP mechanism.

This should be decided after inspecting how much overlap there is between B-cos dynamic multiplication and SP surrogate pullbacks.

## Repository refactor

The B-cos LM repository should be refactored and simplified.

We only need the components relevant to our experiments. In particular:

* keep encoder-only classification models,
* keep explanation and evaluation code,
* keep relevant datasets and metrics,
* remove decoder-only model code,
* remove unused training or experimental utilities where possible,
* make the codebase easier to maintain and extend.

The final repository will likely live under the Cogity organization, but this is still to be decided.

## Transformers dependency

The original B-cos LM code relies heavily on a specific, already old version of `transformers`.

We likely want to update the repository to use a recent version of `transformers`. However, because the code replaces or modifies specific model classes, the exact version should be frozen in `requirements.txt` or equivalent dependency management.

Tasks:

* upgrade the code to a recent `transformers` version,
* verify all patched/replaced model classes,
* freeze the working version in dependencies,
* remove temporary version-specific hacks.

One known issue is in `surrogate_llama.py`, where there is currently code similar to:

```python
if transformers.__version__ == "4.45.2":
    ...
```

This is a temporary hack and should be removed.

## Pullbacks code

The most important pullback-related code is currently in the `pullbacks/` directory.

Some initial work has also been done to adapt parts of the language models to pullbacks, but this was done quickly and should be reviewed carefully.

Tasks:

* inspect the current `pullbacks/` implementation,
* identify which parts can be reused directly from Semantic Pullbacks,
* define the target abstraction for LM pullbacks,
* clean up temporary model-specific patches,
* add tests for the modified backward pass.

## Explainers

There appears to be a bug or design issue in the original B-cos LM code.

The code does not consistently compute `input × gradient` for every explainer. In practice, this seems to happen only inside `BcosExplainer`, indirectly through Captum’s `InputXGradient`.

This should be fixed so that all relevant explainers behave consistently.

Tasks:

* audit all explainers,
* check where `input × gradient` is currently applied,
* make attribution computation explicit,
* ensure consistent behaviour across explainers,
* add tests comparing expected attribution shapes and aggregation behaviour.

## Llama support

The original B-cos LM repository does not include Llama support. Some Llama-related code was added quickly, but it needs to be reviewed and cleaned up.

## VS Code formatting

For local development we use VS Code, Black and isort.

Create or update the following file:

```text
.vscode/settings.json
```

Use this configuration:

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

This makes VS Code format Python files on save and organize imports using the Black-compatible isort profile.



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