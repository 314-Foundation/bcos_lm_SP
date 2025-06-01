import argparse
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from captum.attr import Saliency, DeepLift, GuidedBackprop, InputXGradient, IntegratedGradients, Occlusion, ShapleyValueSampling, DeepLiftShap, GradientShap, KernelShap 
from saliency_utils.lime_utils import explain
from tint.attr import SequentialIntegratedGradients
from datasets import load_dataset
import numpy as np
import json
import os
import random
from tqdm import tqdm
from saliency_utils.utils import batch_loader


class GPTEmbeddingModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(GPTEmbeddingModelWrapper, self).__init__()
        self.model = model

    def forward(self, embeddings, attention_mask=None):
        outputs = self.model(inputs_embeds=embeddings, attention_mask=attention_mask)
        logits = outputs.logits
        logits = logits[:, -1, :]  # Get the last token's logits
        return logits
    
class GPTEmbeddingModelProbWrapper(torch.nn.Module):
    def __init__(self, model):
        super(GPTEmbeddingModelProbWrapper, self).__init__()
        self.model = model

    def forward(self, embeddings, attention_mask=None):
        outputs = self.model(inputs_embeds=embeddings, attention_mask=attention_mask)
        logits = outputs.logits
        logits = logits[:, -1, :]  # Get the last token's logits
        probabilities = torch.softmax(logits, dim=-1)
        return probabilities
    
class GPTModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(GPTModelWrapper, self).__init__()
        self.model = model

    def forward(self, input_ids, attention_mask=None):       
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        logits = logits[:, -1, :]
        return logits
    
class GPTModelProbWrapper(torch.nn.Module):
    def __init__(self, model):
        super(GPTModelProbWrapper, self).__init__()
        self.model = model

    def forward(self, input_ids, attention_mask=None):       
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        logits = logits[:, -1, :]
        probabilities = torch.softmax(logits, dim=-1)
        return probabilities
    
     
class BaseExplainer:

    def _explain(self):
        raise NotImplementedError
    
    def explain(self):
        raise NotImplementedError
    

    def explain_embeddings(self, prefixes, targets, foils, example_indices, contrastive):
        assert len(prefixes) == 1 and len(targets) == 1, "Only one prefix and one target is supported for now"
        inputs = self.tokenizer(prefixes, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        if "position_ids" in inputs:
            position_ids = inputs['position_ids']
        else:
            position_ids = torch.arange(input_ids.size(1), dtype=torch.long, device=self.device).unsqueeze(0).repeat(input_ids.size(0), 1)
        # if targets do not start with a white space, add a white space
        for i in range(len(targets)):
            if targets[i][0] != ' ':
                targets[i] = ' ' + targets[i]
        target_ids = self.tokenizer(targets, return_tensors='pt', add_special_tokens=False)['input_ids'][:, :1]
        target_ids = target_ids.to(self.device)
        if foils is not None:
            for i in range(len(foils)):
                if foils[i][0] != ' ':
                    foils[i] = ' ' + foils[i]
            foil_ids = self.tokenizer(foils, return_tensors='pt', add_special_tokens=False)['input_ids'][:, :1]
            foil_ids = foil_ids.to(self.device)
        explanations = self._explain(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, target_ids=target_ids, foil_ids=foil_ids, example_indices=example_indices, contrastive=contrastive)
        return explanations


    def explain_tokens(self, prefixes, targets, foils, example_indices, contrastive):
        assert len(prefixes) == 1 and len(targets) == 1, "Only one prefix and one target is supported for now"
        inputs = self.tokenizer(prefixes, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        # if targets do not start with a white space, add a white space
        for i in range(len(targets)):
            if targets[i][0] != ' ':
                targets[i] = ' ' + targets[i]
        target_ids = self.tokenizer(targets, return_tensors='pt', add_special_tokens=False)['input_ids'][:, :1]
        target_ids = target_ids.to(self.device)
        if foils is not None:
            for i in range(len(foils)):
                if foils[i][0] != ' ':
                    foils[i] = ' ' + foils[i]
            foil_ids = self.tokenizer(foils, return_tensors='pt', add_special_tokens=False)['input_ids'][:, :1]
            foil_ids = foil_ids.to(self.device)
        explanations = self._explain(input_ids=input_ids, attention_mask=attention_mask, target_ids=target_ids, foil_ids=foil_ids, example_indices=example_indices, contrastive=contrastive)
        return explanations
    
    
    def explain_dataset(self, dataset, contrastive):
        # if class_labels is not provided, then num_classes must be provided
        data_loader = batch_loader(dataset, batch_size=1, shuffle=False)
        saliency_results = {}
        for batch in tqdm(data_loader):
            prefixes = batch['prefix']
            example_indices = batch['index']
            targets = batch['target']
            if 'foil' in batch:
                foils = batch['foil']
            else:
                foils = None
            explanations = self.explain(prefixes=prefixes, targets=targets, foils=foils, example_indices=example_indices, contrastive=contrastive)
            for key, value in explanations.items():
                if key not in saliency_results:
                    saliency_results[key] = []
                saliency_results[key].extend(value)
        return saliency_results
    

class BcosExplainer(BaseExplainer):
    def __init__(self, model, tokenizer):

        self.model = GPTEmbeddingModelWrapper(model)
        self.model.eval()
        self.model.to(model.device)
        self.tokenizer = tokenizer
        self.device = model.device
        #self.explainer = InputXGradient(self.model)
        self.method = "Bcos_absolute"
    
    def _explain(self, input_ids, attention_mask, position_ids=None, target_ids=None, foil_ids=None, example_indices=None, contrastive=False):

        if position_ids is None:
            position_ids = torch.arange(input_ids.size(1), dtype=torch.long, device=self.device).unsqueeze(0).repeat(input_ids.size(0), 1)

        batch_size = input_ids.shape[0]
        assert batch_size == 1, "Batch size must be 1 for now"

        if hasattr(self.model.model, "transformer") and hasattr(self.model.model.transformer, "wte"):
            wte = self.model.model.transformer.wte ## gpt model
        elif hasattr(self.model.model, "model") and hasattr(self.model.model.model, "embed_tokens"):
            wte = self.model.model.model.embed_tokens ## llama model
        else:
            raise ValueError("Model is not supported, cannot extract embeddings")
        embeddings = wte(input_ids) 
        embeddings.requires_grad_()

        # Get the model's predictions
        with torch.no_grad():
            outputs = self.model(embeddings, attention_mask=attention_mask)
        probabilities = torch.softmax(outputs, dim=-1)
        # get the probability of the target token
        target_probabilities = probabilities.gather(1, target_ids) # shape: [batch_size, 1]


        all_saliency_ixg_L1_results = [[] for _ in range(batch_size)]
        all_saliency_ixg_mean_results = [[] for _ in range(batch_size)]


        # activate explanation mode
        with self.model.model.explanation_mode():
            explainer_ixg = InputXGradient(self.model)
            attributions_ixg = explainer_ixg.attribute(
                inputs=(embeddings),
                target=target_ids,
                additional_forward_args=(attention_mask,)
            )

        if contrastive and foil_ids is not None:
            with self.model.model.explanation_mode():
                explainer_ixg = InputXGradient(self.model)
                foil_attributions_ixg = explainer_ixg.attribute(
                    inputs=(embeddings),
                    target=foil_ids,
                    additional_forward_args=(attention_mask,)
                )
            attributions_ixg = attributions_ixg - foil_attributions_ixg

        attributions_ixg_all = attributions_ixg
        for i in range(batch_size):
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[i].detach().cpu().numpy().tolist())
            target_token = self.tokenizer.convert_ids_to_tokens(target_ids[i].detach().cpu().numpy().tolist())[0]                 

            # Compute saliency metrics for each token
            saliency_ixg_L1 = torch.norm(attributions_ixg_all[i:i+1], dim=-1, p=1).detach().cpu().numpy()[0]
            saliency_ixg_mean = attributions_ixg_all[i:i+1].mean(dim=-1).detach().cpu().numpy()[0]
            # Collect results for the current example and class
            # skip padding tokens
            tokens = [token for token in tokens if token != self.tokenizer.pad_token]
            real_length = len(tokens)
            result_ixg_L1 = {
                'index': example_indices[i],
                'prefix': self.tokenizer.decode([t for t in input_ids[i] if not (t in self.tokenizer.all_special_ids and t != self.tokenizer.unk_token_id)], skip_special_tokens=False),
                'target': target_token,
                'target_prob': target_probabilities[i].item(),
                'method': f"{self.method}_ixg_L1",
                'attribution': list(zip(tokens, saliency_ixg_L1.tolist()[:real_length])),
            }

            result_ixg_mean = {
                'index': example_indices[i],
                'prefix': self.tokenizer.decode([t for t in input_ids[i] if not (t in self.tokenizer.all_special_ids and t != self.tokenizer.unk_token_id)], skip_special_tokens=False),
                'target': target_token,
                'target_prob': target_probabilities[i].item(),
                'method': f"{self.method}_ixg_mean",
                "attribution": list(zip(tokens, saliency_ixg_mean.tolist()[:real_length])),
            }
            all_saliency_ixg_L1_results[i].append(result_ixg_L1)
            all_saliency_ixg_mean_results[i].append(result_ixg_mean)

        saliency_results = {f"{self.method}_ixg_mean": all_saliency_ixg_mean_results}
        return saliency_results
    
    def explain(self, prefixes, targets, foils, example_indices, contrastive=False):
        return self.explain_embeddings(prefixes=prefixes, targets=targets, foils=foils, example_indices=example_indices, contrastive=contrastive)
    

class AttentionExplainer(BaseExplainer):
    def __init__(self, model, tokenizer, method=None, baseline='zero'):
        # attention explainer can only explain the predicted classes
        self.model = model
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = model.device

    def _explain(self, input_ids, attention_mask, target_ids, foil_ids, example_indices=None, contrastive=False):

        batch_size = input_ids.shape[0]
        assert batch_size == 1, "Batch size must be 1 for now"

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # compute the probability of the target token
        probabilities = torch.softmax(outputs.logits, dim=-1)[:, -1, :]
        # get the probability of the target token
        target_probabilities = probabilities.gather(1, target_ids)

        attentions = outputs.attentions

        # Stack attentions over layers
        all_attentions = torch.stack(attentions)
        # Get sequence length and batch size
        seq_len = input_ids.shape[1]
        batch_size = input_ids.shape[0]

        # Expand attention mask to match attention shapes
        # Shape: (batch_size, 1, 1, seq_len)
        attention_mask_expanded = attention_mask.unsqueeze(1).unsqueeze(2)

        # Create a mask for attention weights
        # Shape: (batch_size, 1, seq_len, seq_len)
        attention_mask_matrix = attention_mask_expanded * attention_mask_expanded.transpose(-1, -2)

        # Mask out padding tokens in attention weights
        # We set the attention weights corresponding to padding tokens to zero
        all_attentions = all_attentions * attention_mask_matrix.unsqueeze(0)

        # Normalize the attention weights so that they sum to 1 over the real tokens
        # Sum over the last dimension (seq_len)
        attn_weights_sum = all_attentions.sum(dim=-1, keepdim=True) + 1e-9  # Add epsilon to avoid division by zero
        all_attentions = all_attentions / attn_weights_sum

        # Convert input IDs back to tokens
        tokens_batch = [self.tokenizer.convert_ids_to_tokens(ids) for ids in input_ids]

        # Average Attention
        # Average over heads
        avg_attn_heads = all_attentions.mean(dim=2)  # Shape: (num_layers, batch_size, seq_len, seq_len)
        # Average over layers
        avg_attn = avg_attn_heads.mean(dim=0)  # Shape: (batch_size, seq_len, seq_len)

        # Attention Rollout
        rollout = torch.eye(seq_len).unsqueeze(0).repeat(batch_size, 1, 1).to(self.device)  # Shape: (batch_size, seq_len, seq_len)
        for attn in avg_attn_heads:
            attn = attn + torch.eye(seq_len).unsqueeze(0).to(self.device)  # Add identity for self-connections
            attn = attn / attn.sum(dim=-1, keepdim=True)  # Normalize rows
            rollout = torch.bmm(rollout, attn)  # Batch matrix multiplication

        # Extract attention from [CLS] token
        avg_cls_attn = avg_attn[:, -1, :]  # Shape: (batch_size, seq_len)


        all_raw_attention_explanations = []

        # For each example in the batch, print the attention scores
        for i in range(batch_size):
            each_raw_attention_explanations = []
            tokens = tokens_batch[i]                
            valid_len = attention_mask[i].sum().item()  # Number of real tokens
            raw_attention_attribution = avg_cls_attn[i][:int(valid_len)].cpu().numpy()
            target_token = self.tokenizer.convert_ids_to_tokens(target_ids[i].detach().cpu().numpy().tolist())[0]  

            # skip all padding tokens
            raw_attention_result = {
                'index': example_indices[i],
                'prefix': self.tokenizer.decode([t for t in input_ids[i] if not (t in self.tokenizer.all_special_ids and t != self.tokenizer.unk_token_id)], skip_special_tokens=False),
                'target': target_token,
                'target_prob': target_probabilities[i].item(),
                'method': 'raw_attention',
                'attribution': list(zip(tokens[:int(valid_len)], raw_attention_attribution.tolist())),
            }
                
            each_raw_attention_explanations.append(raw_attention_result)
                
        all_raw_attention_explanations.append(each_raw_attention_explanations)
            
        attention_explanations = {"raw_attention": all_raw_attention_explanations}
        return attention_explanations
    
    def explain(self, prefixes, targets, foils, example_indices, contrastive=False):
        return self.explain_tokens(prefixes=prefixes, targets=targets, foils=foils, example_indices=example_indices, contrastive=contrastive)
    
    
class GradientNPropabationExplainer(BaseExplainer):
    def __init__(self, model, tokenizer, method='saliency', baseline='zero'):
        self.model = GPTEmbeddingModelWrapper(model)
        self.model.eval()
        self.model.to(model.device)
        self.tokenizer = tokenizer
        self.method = method
        if method == 'Saliency':
            self.explainer = Saliency(self.model)
        elif method == 'InputXGradient':
            self.explainer = InputXGradient(self.model)
        elif method == 'IntegratedGradients':
            self.explainer = IntegratedGradients(self.model)
        elif method == 'DeepLift':
            self.explainer = DeepLift(self.model)
        elif method == 'GuidedBackprop':
            self.explainer = GuidedBackprop(self.model)
        elif method == 'SIG':
            self.explainer = SequentialIntegratedGradients(self.model)
        else:
            raise ValueError(f"Invalid method {method}")
        self.device = model.device
        if baseline == 'zero':
            self.baseline = None
        elif baseline == 'mask':
            self.baseline = self.tokenizer.mask_token_id
        elif baseline == 'pad':
            self.baseline = self.tokenizer.pad_token_id
        else:
            raise ValueError(f"Invalid baseline {baseline}")

    def _explain(self, input_ids, attention_mask, position_ids=None, target_ids=None, foil_ids=None, example_indices=None, contrastive=False):

        if position_ids is None:
            position_ids = torch.arange(input_ids.size(1), dtype=torch.long, device=self.device).unsqueeze(0).repeat(input_ids.size(0), 1)
        

        batch_size = input_ids.shape[0]
        assert batch_size == 1, "Batch size must be 1 for now"

        if hasattr(self.model.model, "transformer") and hasattr(self.model.model.transformer, "wte"):
            wte = self.model.model.transformer.wte ## gpt model
        elif hasattr(self.model.model, "model") and hasattr(self.model.model.model, "embed_tokens"):
            wte = self.model.model.model.embed_tokens ## llama model
        else:
            raise ValueError("Model is not supported, cannot extract embeddings")
        #wpe = self.model.model.transformer.wpe
        embeddings = wte(input_ids) 
        embeddings.requires_grad_()

        # Get the model's predictions
        with torch.no_grad():
            outputs = self.model(embeddings, attention_mask=attention_mask)
        probabilities = torch.softmax(outputs, dim=-1)
        # get the probability of the target token
        target_probabilities = probabilities.gather(1, target_ids) # shape: [batch_size, 1]

        all_saliency_L1_results = [[] for _ in range(batch_size)]
        all_saliency_mean_results = [[] for _ in range(batch_size)]
        
        if self.method == 'Saliency':
            attributions = self.explainer.attribute(
                inputs=(embeddings),
                target=target_ids,
                additional_forward_args=(attention_mask,),
                abs=False,
            )
        elif self.method == 'IntegratedGradients' or self.method == 'DeepLift' or self.method == 'SIG':
            if self.baseline is not None:
                token_baseline_ids = torch.ones_like(input_ids) * self.baseline 
                baselines = wte(token_baseline_ids)
            else:
                baselines = None
            attributions = self.explainer.attribute(
                inputs=(embeddings),
                baselines=baselines,
                target=target_ids,
                additional_forward_args=(attention_mask,)
            )
        else:
            attributions = self.explainer.attribute(
                inputs=(embeddings),
                target=target_ids,
                additional_forward_args=(attention_mask,)
            )

        if contrastive and foil_ids is not None:
            if self.method == 'Saliency':
                foil_attributions = self.explainer.attribute(
                    inputs=(embeddings),
                    target=foil_ids,
                    additional_forward_args=(attention_mask,),
                    abs=False,
                )
            elif self.method == 'IntegratedGradients' or self.method == 'DeepLift' or self.method == 'SIG':
                if self.baseline is not None:
                    token_baseline_ids = torch.ones_like(input_ids) * self.baseline 
                    baselines = wte(token_baseline_ids)
                else:
                    baselines = None
                foil_attributions = self.explainer.attribute(
                    inputs=(embeddings),
                    baselines=baselines,
                    target=foil_ids,
                    additional_forward_args=(attention_mask,)
                )
            else:
                foil_attributions = self.explainer.attribute(
                    inputs=(embeddings),
                    target=foil_ids,
                    additional_forward_args=(attention_mask,)
                )
            attributions = attributions - foil_attributions
   
        attributions_all = attributions


        for i in range(batch_size):

            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[i].detach().cpu().numpy().tolist())
            target_token = self.tokenizer.convert_ids_to_tokens(target_ids[i].detach().cpu().numpy().tolist())[0]                  

            # Compute saliency metrics for each token
            saliency_L1 = torch.norm(attributions_all[i:i+1], dim=-1, p=1).detach().cpu().numpy()[0]
            saliency_mean = attributions_all[i:i+1].mean(dim=-1).detach().cpu().numpy()[0]
            # Collect results for the current example and class
            # skip padding tokens
            tokens = [token for token in tokens if token != self.tokenizer.pad_token]
            real_length = len(tokens)
            result_L1 = {
                'index': example_indices[i],
                'prefix': self.tokenizer.decode([t for t in input_ids[i] if not (t in self.tokenizer.all_special_ids and t != self.tokenizer.unk_token_id)], skip_special_tokens=False),
                'target': target_token,
                'target_prob': target_probabilities[i].item(),
                'method': f"{self.method}_L1",
                'attribution': list(zip(tokens, saliency_L1.tolist()[:real_length])),
            }

            result_mean = {
                'index': example_indices[i],
                'prefix': self.tokenizer.decode([t for t in input_ids[i] if not (t in self.tokenizer.all_special_ids and t != self.tokenizer.unk_token_id)], skip_special_tokens=False),
                'target': target_token,
                'target_prob': target_probabilities[i].item(),
                'method': f"{self.method}_mean",
                "attribution": list(zip(tokens, saliency_mean.tolist()[:real_length])),
            }
            all_saliency_L1_results[i].append(result_L1)
            all_saliency_mean_results[i].append(result_mean)
        saliency_results = {f"{self.method}_L1": all_saliency_L1_results, f"{self.method}_mean": all_saliency_mean_results}
        return saliency_results
    
    def explain(self, prefixes, targets, foils, example_indices, contrastive=False):
        return self.explain_embeddings(prefixes=prefixes, targets=targets, foils=foils, example_indices=example_indices, contrastive=contrastive)
    
    
class OcclusionExplainer(BaseExplainer):
    def __init__(self, model, tokenizer, method='Occlusion', baseline='pad'):
        self.model = GPTModelProbWrapper(model)
        self.model.eval()
        self.model.to(model.device)
        self.tokenizer = tokenizer
        self.explainer = Occlusion(self.model)
        # Occlusion parameters
        self.sliding_window_size = (1,)  # Occlude one token at a time
        if baseline == 'zero':
            self.baseline = None
        elif baseline == 'mask':
            self.baseline = self.tokenizer.mask_token_id
        elif baseline == 'pad':
            self.baseline = self.tokenizer.pad_token_id
        else:
            raise ValueError(f"Invalid baseline {baseline}")

        self.stride = (1,)
        self.device = model.device

    def _explain(self, input_ids, attention_mask, target_ids, foil_ids, example_indices, contrastive):

        batch_size = input_ids.shape[0]
        assert batch_size == 1, "Batch size must be 1 for now"

        # Get the model's predictions
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        probabilities = outputs
        # get the probability of the target token
        target_probabilities = probabilities.gather(1, target_ids) # shape: [batch_size, 1]
        

        all_occlusion_results = [[] for _ in range(batch_size)]
        
        attributions = self.explainer.attribute(
            inputs=input_ids,
            strides=self.stride,
            sliding_window_shapes=self.sliding_window_size,
            baselines=self.baseline,
            target=target_ids,
            additional_forward_args=(attention_mask,)
        )

        if contrastive and foil_ids is not None:
            foil_attributions = self.explainer.attribute(
                inputs=input_ids,
                strides=self.stride,
                sliding_window_shapes=self.sliding_window_size,
                baselines=self.baseline,
                target=foil_ids,
                additional_forward_args=(attention_mask,)
            )
            attributions = attributions - foil_attributions

        for i in range(batch_size):
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[i].detach().cpu().numpy().tolist())
            target_token = self.tokenizer.convert_ids_to_tokens(target_ids[i].detach().cpu().numpy().tolist())[0]
            attributions_i = attributions.detach().cpu().numpy()[i]  # Shape: [seq_len]
            # skip padding tokens
            tokens = [token for token in tokens if token != self.tokenizer.pad_token]
            real_length = len(tokens)
            # Collect results for the current example and class
            result = {
                'index': example_indices[i],
                'prefix': self.tokenizer.decode([t for t in input_ids[i] if not (t in self.tokenizer.all_special_ids and t != self.tokenizer.unk_token_id)], skip_special_tokens=False),
                'target': target_token,
                'target_prob': target_probabilities[i].item(),
                'method': 'Occlusion',
                'attribution': list(zip(tokens, attributions_i.tolist()[:real_length])),
            }
            all_occlusion_results[i].append(result)
        return {"Occlusion": all_occlusion_results}
    
    def explain(self, prefixes, targets, foils, example_indices, contrastive=False):
        return self.explain_tokens(prefixes=prefixes, targets=targets, foils=foils, example_indices=example_indices, contrastive=contrastive)
    
    
