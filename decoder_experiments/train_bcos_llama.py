from bcos_lm.models.modeling_llama import LlamaForCausalLM
from transformers.trainer_utils import EvalLoopOutput
from transformers import AutoTokenizer, AutoConfig
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, TrainerCallback
import transformers
import torch
import time
import argparse
import logging
import random
from datasets import load_dataset
# import glob
# import csv
# import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
from typing import Dict, Any
# from tqdm.notebook import tqdm
# tqdm.pandas()
import torch.distributed as dist
from accelerate import infer_auto_device_map, dispatch_model, init_empty_weights, load_checkpoint_and_dispatch


def main():
    # Argument parser for hyperparameters
    parser = argparse.ArgumentParser(description="Fine-tune BERT for sequence classification")


    # Hyperparameters
    parser.add_argument('--model_name_or_path', type=str, default='bert-base-uncased',
                        help='Pre-trained model name or path')
    parser.add_argument('--dataset_name', type=str, default='fancyzhx/ag_news',
                        help='Dataset name (default: ag_news)')
    parser.add_argument('--num_labels', type=int, default=4,
                        help='Number of labels in the dataset')
    parser.add_argument('--output_dir', type=str, default='bcos_bert_base_agnews_512',
                        help='Directory to save the model')
    parser.add_argument('--max_seq_length', type=int, default=512,
                        help='Maximum input sequence length after tokenization')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training and evaluation')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of updates steps to accumulate before performing a backward/update pass')
    parser.add_argument('--learning_rate', type=float, default=3e-5,
                        help='Learning rate for the optimizer')
    parser.add_argument('--warmup_steps_or_ratio', type=float, default=0.1,
                        help='Number or ratio of warmup steps for the learning rate scheduler')
    parser.add_argument('--num_train_epochs', type=int, default=10,
                        help='Total number of training epochs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for initialization')
    parser.add_argument('--early_stopping_patience', type=int, default=-1,
                        help='Number of epochs with no improvement after which training will be stopped')
    parser.add_argument('--log_file', type=str, default='training.log',
                        help='Path to the log file')
    parser.add_argument('--eval_steps', type=int, default=2000,
                        help='Evaluate the model every X training steps')
    parser.add_argument('--save_steps', type=int, default=2000,
                        help='Save the model every X training steps')
    parser.add_argument("--num_train_examples", type=int, default=1000000,)
    parser.add_argument("--num_eval_examples", type=int, default=10000,)
    parser.add_argument('--b', type=float, default=2.0,)
    parser.add_argument('--bcos', action='store_true', help='Use Bcos')
    parser.add_argument('--bce', action='store_true', help='Use bce loss instead of cross entropy loss')
    parser.add_argument("--bcos_lm_head", action='store_true', help="Use bcos lm head")



    args = parser.parse_args()
    print("start experiment")

    
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    
    # create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
   
    log_file = os.path.join(args.output_dir, args.log_file)


    # Set up logging
    logging.basicConfig(
        filename=log_file,
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        force=True
    )
    transformers.logging.set_verbosity_info()
    # Also log to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


    # Log the hyperparameters
    logging.info("Hyperparameters:")
    for arg in vars(args):
        logging.info(f"{arg}: {getattr(args, arg)}")


    # Set up the device for GPU usage if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")


    # Set seeds for reproducibility
    seed_val = args.seed


    def set_random_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    set_random_seed(seed_val+dist.get_rank())
    #set_random_seed(seed_val)

    # Load the dataset

    logging.info(f"Loading {args.dataset_name} dataset...")

    if "webtext" in args.dataset_name:
        dataset = load_dataset("Skylion007/openwebtext", trust_remote_code=True)['train']
        len_dataset = len(dataset)
        num_val = args.num_eval_examples if args.num_eval_examples > 0 else 10000
        num_train = args.num_train_examples if args.num_train_examples > 0 else len_dataset - num_val
        num_test = 10000
        train_dataset = dataset.select(range(num_train))
        val_dataset = dataset.select(range(len_dataset - num_val - num_test, len_dataset - num_test))
        test_dataset = dataset.select(range(len_dataset - num_test, len_dataset))
  
    else:
        print("Only webtext dataset is supported for now")
    
    # Create Masked Language Model
    print("dataset loaded")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(examples['text'],
                         padding='max_length',
                         truncation=True,
                         max_length=args.max_seq_length,
                         return_tensors='pt'
                         )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )


    # Initialize the tokenizer and model
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    config.bcos = args.bcos
    config.b = args.b
    config.bce = args.bce
    config.bcos_lm_head = args.bcos_lm_head

    model = LlamaForCausalLM.load_from_pretrained(args.model_name_or_path, config=config, device_map=None)
    # print usable gpus
    print("usable gpus: ", torch.cuda.device_count())
  
    model.to(device)
    print("model loaded")
    # Create lamda tokenizing function

    # Apply tokenization to the datasets
    logging.info("Tokenizing datasets...")
    tokenized_train_datasets = train_dataset.map(tokenize_function, batched=True, num_proc=8)


    # Set the format of the datasets to PyTorch tensors
    tokenized_train_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    tokenized_eval_datasets = val_dataset.map(tokenize_function, batched=True, num_proc=8)
    tokenized_eval_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask'])



    def compute_metrics(eval_preds, 
        ):
        """
        Compute the cross entropy loss
        """
        # clear cuda cache
        #if compute_results:
        print("computing metrics")
        logits, labels = eval_preds
        logits = torch.tensor(logits, device="cpu")
        labels = torch.tensor(labels, device="cpu")
        shift_logits = torch.tensor(logits[:, :-1, :])
        shift_labels = torch.tensor(labels[:, 1:])
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
    
        return {"cross_entropy": loss.item()}
    
    
    # Compute the number of warmup steps

    warmup_steps = int(args.warmup_steps_or_ratio * len(train_dataset) // (args.batch_size * args.gradient_accumulation_steps) * args.num_train_epochs)
    print("start training")

    if args.bce:
        batch_eval_metrics = True
    else:
        batch_eval_metrics = False

    if args.bce:
        metric_for_best_model = "cross_entropy"
    else:
        metric_for_best_model = "loss"

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=1,
        eval_steps=args.eval_steps,
        load_best_model_at_end=True,    
        metric_for_best_model=metric_for_best_model,
        greater_is_better=False,
        logging_steps=args.eval_steps,
        log_level="info",
        logging_dir=os.path.join(args.output_dir, "logs"),
        prediction_loss_only=False,
        learning_rate=args.learning_rate,
        lr_scheduler_type="linear",
        warmup_steps=warmup_steps,
        batch_eval_metrics=batch_eval_metrics,
        gradient_checkpointing=True,
        fp16=False,
        optim="adafactor",
    )

    class CrossEntropyAggregator:
        """
        Maintains a running sum of negative log-likelihood and count of samples
        so we can compute cross-entropy across all batches so far.
        """
        def __init__(self):
            self.total_neg_log_likelihood = 0.0
            self.total_count = 0

        def update(self, logits: np.ndarray, labels: np.ndarray):
            """
            Update the running sums with the new batch logits and labels.
            """
            # shift logits by their max for numerical stability
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
            # add the loss of all valid positions
            count = (shift_labels != -100).sum().item()
            self.total_neg_log_likelihood += loss.item() * count
            self.total_count += count
            

        def compute(self) -> float:
            """Compute the average cross-entropy across all data so far."""
            if self.total_count == 0:
                return 0.0
            return self.total_neg_log_likelihood / self.total_count

        def reset(self):
            """Optional: reset the aggregator if needed."""
            self.total_neg_log_likelihood = 0.0
            self.total_count = 0

    #
    # 2) Instantiate a global aggregator
    #
    cross_entropy_aggregator = CrossEntropyAggregator()

    #
    # 3) Define the per-batch compute_metrics function
    #
    def compute_metrics(eval_pred, compute_result=False):
        """
        Called once per batch if 'batch_eval_metrics=True' in your custom Trainer.
        eval_pred: (logits, labels) for the *current batch*.
        """
        logits, labels = eval_pred
        if not compute_result:        
            cross_entropy_aggregator.update(logits, labels)
            cross_entropy_loss = cross_entropy_aggregator.compute()
            return {"cross_entropy": cross_entropy_loss}
        else:
            cross_entropy_aggregator.update(logits, labels)
            cross_entropy_loss = cross_entropy_aggregator.compute()
            cross_entropy_aggregator.reset()
            return {"cross_entropy": cross_entropy_loss}


    if args.bce:
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=tokenized_train_datasets,
            eval_dataset=tokenized_eval_datasets,
            compute_metrics=compute_metrics,
        )
    else: 
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=tokenized_train_datasets,
            eval_dataset=tokenized_eval_datasets,
        )


    trainer.train()


    # 3. Identify the best checkpoint (Trainer tracked it during training)
    best_checkpoint = trainer.state.best_model_checkpoint
    print("Best checkpoint is:", best_checkpoint)
    #print("Best step:", trainer.state.best_step)
    logging.info(f"Best checkpoint is: {best_checkpoint}")
    #logging.info(f"Best step: {trainer.state.best_step}")

    # copy everything from the best checkpoint to the output directory
    os.system(f"cp -r {best_checkpoint}/* {args.output_dir}")

    # save config
    config.save_pretrained(args.output_dir)
    # save tokenizer
    tokenizer.save_pretrained(args.output_dir)



if __name__ == "__main__":
    main()





