import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from bcos_lm.models.modeling_gpt2 import GPT2LMHeadModel
from bcos_lm.models.modeling_llama import LlamaForCausalLM
from saliency_utils.Explainer_decoder import GradientNPropabationExplainer, OcclusionExplainer, BcosExplainer
import pandas as pd
import os
import json

DATA_DIR = "decoder_experiments/data"

blimp_datasets = ["anaphor_gender_agreement",
            "anaphor_number_agreement",
            "animate_subject_passive",
            "determiner_noun_agreement_1",
            "determiner_noun_agreement_irregular_1",
            "determiner_noun_agreement_with_adjective_1",
            "determiner_noun_agreement_with_adj_irregular_1",
            "npi_present_1",
            "distractor_agreement_relational_noun",
            #"irregular_plural_subject_verb_agreement_1",
            #"regular_plural_subject_verb_agreement_1",
            ]
        
ioi_datasets = ["ioi_dataset"]
sva_datasets = ["lgd_dataset"]

datasets = {"blimp": blimp_datasets,
            "ioi": ioi_datasets,
            }

EXPLANATION_METHODS = {
    "Bcos": BcosExplainer,
    "Saliency": GradientNPropabationExplainer,
    "InputXGradient": GradientNPropabationExplainer,  
    "Occlusion": OcclusionExplainer,
}

# load dataset
def load_dataset(dataset_type, dataset_name):
    dataset = {}
    data_path = f"{DATA_DIR}/{dataset_type}_with_targets/{dataset_name}.csv"
    # load the pandas dataframe
    df = pd.read_csv(data_path)
    one_sentence_prefixes = df['one_prefix_prefix'].tolist()
    one_word_targets = df['one_prefix_word_good'].tolist()
    one_word_foils = df['one_prefix_word_bad'].tolist()
    if 'target_phrase' in df.columns:
        evidences = df['target_phrase'].tolist()
    else:
        evidences = df['target'].tolist()
    indexes = list(range(len(one_sentence_prefixes)))
    dataset['prefix'] = one_sentence_prefixes
    dataset['target'] = one_word_targets
    dataset['foil'] = one_word_foils
    dataset['evidence'] = evidences
    dataset['index'] = indexes
    return dataset

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="gpt2")
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()
    # Load the model and tokenizer
    model_dir = args.model_dir
    model_name_or_path = "gpt2"
    config = AutoConfig.from_pretrained(model_dir)
    if "gpt" in model_dir:
        model = GPT2LMHeadModel.load_from_pretrained(model_dir, config=config)
    elif "llama" in model_dir:
        model = LlamaForCausalLM.load_from_pretrained(model_dir, config=config)
    else:
        raise ValueError(f"Model {model_dir} not supported.")
    #model = GPT2LMHeadModel.from_pretrained(model_name, output_attentions=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    results_dir = f"decoder_experiments/{args.output_dir}/contrastive_explanations"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Load the dataset
    for data_type in datasets.keys():
        if not os.path.exists(os.path.join(results_dir, data_type)):
            os.makedirs(os.path.join(results_dir, data_type))
        for dataset_name in datasets[data_type]:
            dataset = load_dataset(data_type, dataset_name)
            print(f"Loaded {dataset_name} dataset with {len(dataset['prefix'])} examples.")

            for method in EXPLANATION_METHODS.keys():
                print(f"Generating explanations using {method} method...")
                
                if EXPLANATION_METHODS[method] == BcosExplainer:
                    explainer = BcosExplainer(model, tokenizer)
                # for GradientNPropabationExplainer, we need to specify the method
                elif EXPLANATION_METHODS[method] == GradientNPropabationExplainer:
                    explainer = EXPLANATION_METHODS[method](model, tokenizer, method, 'pad')
                else:
                    explainer = EXPLANATION_METHODS[method](model, tokenizer, baseline='pad') 

                explanation_results = explainer.explain_dataset(dataset, contrastive=True)
    
                # Save the explanations to a file
                output_file = f"{results_dir}/{data_type}/{dataset_name}_{method}_explanations.json"
                with open(output_file, 'w') as f:
                    json.dump(explanation_results, f, indent=4)
                print(f"Saved explanations to {output_file}")
            print(f"Finished processing {dataset_name} dataset.")