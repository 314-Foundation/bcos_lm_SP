#!/usr/bin/env bash

# bcos_llama
torchrun --nproc_per_node 4 -m decoder_experiments.train_bcos_llama \
    --model_name_or_path meta-llama/Llama-3.2-1B \
    --dataset_name="webtext" \
    --output_dir "bcos_llama" \
    --batch_size=32 \
    --gradient_accumulation_steps=1 \
    --max_seq_length=1024 \
    --learning_rate=5e-04 \
    --num_train_epochs=1 \
    --seed=42 \
    --b 1.1 \
    --bcos \
    --warmup_steps_or_ratio=0.01 \
    --num_train_examples=4000000 \
    --num_eval_examples=10000 \


python -m gen_contrastive_explanations --model_dir "bcos_llama" --output_dir "llama_results"
python -m eval_explanations --explanation_dir "llama_results"
python -m compute_prob_gap --model_dir "bcos_llama" --output_dir "llama_results"
python -m compute_prob_gap --model_dir "meta-llama/Llama-3.2-1B" --output_dir "vanilla_llama_results"

# bcos_gpt2
python -m decoder_experiments.train_bcos_gpt2 \
    --model_name_or_path gpt2 \
    --dataset_name="webtext" \
    --output_dir "bcos_gpt2" \
    --batch_size=16 \
    --gradient_accumulation_steps=1 \
    --max_seq_length=512 \
    --learning_rate=5e-04 \
    --num_train_epochs=1 \
    --warmup_steps_or_ratio=0.01 \
    --num_train_examples=500000 \
    --num_eval_examples=10000 \
    --seed=42 \
    --b 1.1 \
    --bcos \

python -m gen_contrastive_explanations --model_dir "bcos_gpt2" --output_dir "gpt2_results"
python -m eval_explanations --explanation_dir "gpt2_results"
python -m compute_prob_gap --model_dir "bcos_gpt2" --output_dir "gpt2_results"
python -m compute_prob_gap --model_dir "gpt2" --output_dir "vanilla_gpt2_results"
