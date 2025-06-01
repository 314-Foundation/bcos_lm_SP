import torch
from transformers import AutoTokenizer
import pandas as pd
import os
import json
import random
from math import comb

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
    "Bcos",
    "Saliency",
    "InputXGradient",
    "Occlusion",
}

# load dataset
def load_dataset(dataset_type, dataset_name):
    dataset = {}
    data_path = f"{DATA_DIR}/{dataset_type}_with_targets/{dataset_name}.csv"
    # load the pandas dataframe
    df = pd.read_csv(data_path)
    one_sentence_prefixes = df['one_prefix_prefix'].tolist()
    one_word_targets = df['one_prefix_word_good'].tolist()
    if dataset_name != 'distractor_agreement_relational_noun' and dataset_name != 'irregular_plural_subject_verb_agreement_1' and dataset_name != 'regular_plural_subject_verb_agreement_1' and data_type != 'sva':
        if 'target_phrase' in df.columns:
            evidences = df['target_phrase'].tolist()
        else:
            evidences = df['target'].tolist()
    else:
        evidences = df['target'].tolist()
    indexes = list(range(len(one_sentence_prefixes)))
    dataset['prefix'] = one_sentence_prefixes
    dataset['target'] = one_word_targets
    dataset['evidence'] = evidences
    dataset['index'] = indexes
    return dataset

def get_evidence_ids(prefix, evidence, tokenizer):
    # Get the evidence ids from the prefix and evidence

    # if evidence is nan
    if not isinstance(evidence, str):
        return None

    if evidence == "nan":
        return None
    
    if len(evidence) >= len(prefix):
        return None
    

    prefix_token_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    evidence_token_ids_no_white_space = tokenizer(evidence, add_special_tokens=False)["input_ids"]
    evidence_token_ids = tokenizer(" " + evidence, add_special_tokens=False)["input_ids"]

    # try to find the evidence in the prefix
    evidence_ids = []
    for i in range(len(prefix_token_ids) - len(evidence_token_ids) + 1):
        if prefix_token_ids[i:i+len(evidence_token_ids)] == evidence_token_ids:
            evidence_ids.append(list(range(i, i+len(evidence_token_ids))))
            
    for i in range(len(prefix_token_ids) - len(evidence_token_ids_no_white_space) + 1):
        if prefix_token_ids[i:i+len(evidence_token_ids_no_white_space)] == evidence_token_ids_no_white_space:
            evidence_ids.append(list(range(i, i+len(evidence_token_ids_no_white_space))))
    
    if len(evidence_ids) == 0:
        #print(f"Evidence {evidence} not found in prefix {prefix}")
        return None
    else:
        evidence_ids = [item for sublist in evidence_ids for item in sublist]
        evidence_ids = list(set(evidence_ids))
        evidence_ids.sort()
        return evidence_ids

def expected_inverse_of_min(m: int, n: int) -> float:

    if not (1 <= n < m):
        raise ValueError("n must satisfy 1 <= n < m")

    total_ways = comb(m, n)
    numerator_sum = 0.0
    for k in range(1, m - n + 2):
        numerator_sum += comb(m - k, n - 1) / k

    return numerator_sum / total_ways
   
def measure_random_mrr(attribution, evidence_ids):
    if evidence_ids is None:
        return None
    else:
        prefix_len = len(attribution)
        evidence_len = len(evidence_ids)
        # Calculate the expected inverse of the maximum
        expected_value = expected_inverse_of_min(prefix_len, evidence_len)
        return expected_value
    
    
def measure_mrr(attribution, evidence_ids):
    if evidence_ids is None:
        return None
    attribution_scores = [attr[1] for attr in attribution]
    # if attribution scores are all 0, return none
    if all(score == 0.0 for score in attribution_scores):
        return None
    # Sort the attribution scores in descending order, find the smallest index of any id in the evidence ids
    sorted_indices = sorted(range(len(attribution_scores)), key=lambda k: attribution_scores[k], reverse=True)
    evidence_mrr = []
    for i in range(len(sorted_indices)):
        if sorted_indices[i] in evidence_ids:
            evidence_mrr.append(1/(i+1))
    # return the largest mrr
    if len(evidence_mrr) == 0:
        return None
    else:
        return max(evidence_mrr)
    
def evaluate_dataset(dataset_type, dataset_name, tokenizer, explanation_dir):
    bos_token = tokenizer.bos_token
    print(bos_token)
    dataset = load_dataset(dataset_type, dataset_name)
    output_dir = explanation_dir.replace("contrastive_explanations", "contrastive_scores")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = f"{output_dir}/{dataset_type}/{dataset_name}_mrr.json"
    if not os.path.exists(os.path.join(output_dir, dataset_type)):
        os.makedirs(os.path.join(output_dir, dataset_type))
    prefixes = dataset['prefix']
    evidences = dataset['evidence']
    evidence_ids = [get_evidence_ids(prefix, evidence, tokenizer) for prefix, evidence in zip(prefixes, evidences)]
    num_valid_evidence_ids = sum([1 for evidence_id in evidence_ids if evidence_id is not None])
    mrr_results = {"num_examples": num_valid_evidence_ids, "MRR": {}}
    random_results = []
    for method in EXPLANATION_METHODS:
        explanation_path = f"{explanation_dir}/{dataset_type}/{dataset_name}_{method}_explanations.json"
        if not os.path.exists(explanation_path):
            print(f"Explanation file {explanation_path} does not exist. Skipping {method} method.")
            continue
        with open(explanation_path, 'r') as f:
            outputs = json.load(f)  
        attribution_methods = outputs.keys()
        for attribution_method in attribution_methods:
            if attribution_method == "Saliency_mean" or attribution_method == "InputXGradient_L1":
                continue
            explanations = outputs[attribution_method]
            mrrs = []
            assert len(explanations) == len(dataset['prefix']), f"Length of explanations {len(explanations)} does not match length of dataset {len(dataset['prefix'])}"
            for i in range(len(explanations)):
                attribution = explanations[i][0]["attribution"]
                if attribution[0][0] == bos_token:
                    attribution = attribution[1:]
                evidence_id = evidence_ids[i]
                mrr = measure_mrr(attribution, evidence_id)
                if "random" not in mrr_results["MRR"].keys():
                    random_mrr = measure_random_mrr(attribution, evidence_id)
                    if random_mrr is not None:
                        random_results.append(random_mrr)
                if mrr is not None:
                    mrrs.append(mrr)
            print(f"Number of valid evidence ids for {dataset_name} with {attribution_method}: {len(mrrs)}")
            if "random" not in mrr_results["MRR"].keys():
                if len(random_results) == 0:
                    mean_random_mrr = -1
                else:   
                    mean_random_mrr = sum(random_results) / len(random_results)
                mrr_results["MRR"]["random"] = mean_random_mrr
            if len(mrrs) == 0:
                mean_mrr = -1
            else:
                mean_mrr = sum(mrrs) / len(mrrs)
            mrr_results["MRR"][f"{attribution_method}"] = mean_mrr


    with open(output_path, 'w') as f:
        json.dump(mrr_results, f, indent=4)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--explanation_dir", type=str, help="Directory where the explanation files are stored.")
    args = parser.parse_args()
    random.seed(42)
    # Load the model and tokenizer
    if "llama" in args.explanation_dir:
        model_name = "meta-llama/Llama-3.2-1B"
    elif "gpt" in args.explanation_dir:
        model_name = "gpt2"
    else:
        raise ValueError("Model name not found in explanation directory. Please check the directory name.")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    explanation_dir = f"decoder_experiments/{args.explanation_dir}/contrastive_explanations"
    
    if not os.path.exists(explanation_dir):
        raise ValueError(f"Results directory {explanation_dir} does not exist. Please run the explanation generation script first.")
    
    # Load the dataset
    for data_type in datasets.keys():
        if not os.path.exists(os.path.join(explanation_dir, data_type)):
            raise ValueError(f"Results directory for {data_type} does not exist. Please run the explanation generation script first.")
        for dataset_name in datasets[data_type]:
            evaluate_dataset(data_type, dataset_name, tokenizer, explanation_dir)
            print(f"Evaluated {dataset_name} dataset.")