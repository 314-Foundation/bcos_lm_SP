import torch
import transformers
from torch import nn
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    apply_rotary_pos_emb,
    repeat_kv,
)

from .surrogate_module import SurrogateModule


class SurrogateLlamaRMSNorm(SurrogateModule):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # return self.weight * hidden_states.to(input_dtype)

        mult = torch.rsqrt(variance + self.variance_epsilon) * self.weight
        mult = mult.to(input_dtype)
        mult = (
            mult.detach()
        )  # detach the multiplier to prevent gradients from flowing through it

        return hidden_states.to(input_dtype) * mult

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


if transformers.__version__ == "4.45.2":
    import math

    import torch.nn.functional as F

    class SurrogateLlamaAttention(SurrogateModule, LlamaAttention):
        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=None,
            position_embeddings=None,  # will become mandatory in v4.46
            **kwargs,
        ):
            bsz, q_len, _ = hidden_states.size()

            if self.config.pretraining_tp > 1:
                key_value_slicing = (
                    self.num_key_value_heads * self.head_dim
                ) // self.config.pretraining_tp
                query_slices = self.q_proj.weight.split(
                    (self.num_heads * self.head_dim) // self.config.pretraining_tp,
                    dim=0,
                )
                key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
                value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

                query_states = [
                    F.linear(hidden_states, query_slices[i])
                    for i in range(self.config.pretraining_tp)
                ]
                query_states = torch.cat(query_states, dim=-1)

                key_states = [
                    F.linear(hidden_states, key_slices[i])
                    for i in range(self.config.pretraining_tp)
                ]
                key_states = torch.cat(key_states, dim=-1)

                value_states = [
                    F.linear(hidden_states, value_slices[i])
                    for i in range(self.config.pretraining_tp)
                ]
                value_states = torch.cat(value_states, dim=-1)

            else:
                query_states = self.q_proj(hidden_states)
                key_states = self.k_proj(hidden_states)
                value_states = self.v_proj(hidden_states)

            query_states = query_states.view(
                bsz, q_len, self.num_heads, self.head_dim
            ).transpose(1, 2)
            key_states = key_states.view(
                bsz, q_len, self.num_key_value_heads, self.head_dim
            ).transpose(1, 2)
            value_states = value_states.view(
                bsz, q_len, self.num_key_value_heads, self.head_dim
            ).transpose(1, 2)

            if position_embeddings is None:
                # logger.warning_once(
                #     "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                #     "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                #     "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                #     "removed and `position_embeddings` will be mandatory."
                # )
                cos, sin = self.rotary_emb(value_states, position_ids)
            else:
                cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

            if past_key_value is not None:
                # sin and cos are specific to RoPE models; cache_position needed for the static cache
                cache_kwargs = {
                    "sin": sin,
                    "cos": cos,
                    "cache_position": cache_position,
                }
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )

            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)
            attn_weights = torch.matmul(
                query_states, key_states.transpose(2, 3)
            ) / math.sqrt(self.head_dim)

            if attention_mask is not None:  # no matter the length, we just slice it
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(query_states.dtype)
            attn_weights = nn.functional.dropout(
                attn_weights, p=self.attention_dropout, training=self.training
            )
            if not self.standard_backward:
                attn_weights = attn_weights.detach()

            attn_output = torch.matmul(attn_weights, value_states)

            if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )

            attn_output = attn_output.transpose(1, 2).contiguous()

            attn_output = attn_output.reshape(bsz, q_len, -1)

            if self.config.pretraining_tp > 1:
                attn_output = attn_output.split(
                    self.hidden_size // self.config.pretraining_tp, dim=2
                )
                o_proj_slices = self.o_proj.weight.split(
                    self.hidden_size // self.config.pretraining_tp, dim=1
                )
                attn_output = sum(
                    [
                        F.linear(attn_output[i], o_proj_slices[i])
                        for i in range(self.config.pretraining_tp)
                    ]
                )
            else:
                attn_output = self.o_proj(attn_output)

            if not output_attentions:
                attn_weights = None

            return attn_output, attn_weights, past_key_value

else:
    print("You are using a version of transformers that is not 4.45.2!")

    # def eager_attention_forward(
    #     module: nn.Module,
    #     query: torch.Tensor,
    #     key: torch.Tensor,
    #     value: torch.Tensor,
    #     attention_mask: torch.Tensor | None,
    #     scaling: float,
    #     dropout: float = 0.0,
    #     **kwargs,
    # ):
    #     key_states = repeat_kv(key, module.num_key_value_groups)
    #     value_states = repeat_kv(value, module.num_key_value_groups)

    #     standard_backward = kwargs.get("standard_backward", True)
    #     # TODO - add temperature scaling here if needed

    #     attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    #     if attention_mask is not None:
    #         attn_weights = attn_weights + attention_mask

    #     attn_weights = nn.functional.softmax(
    #         attn_weights, dim=-1, dtype=torch.float32
    #     ).to(query.dtype)
    #     attn_weights = nn.functional.dropout(
    #         attn_weights, p=dropout, training=module.training
    #     )
    #     if not standard_backward:
    #         attn_weights = attn_weights.detach()

    #     attn_output = torch.matmul(attn_weights, value_states)
    #     attn_output = attn_output.transpose(1, 2).contiguous()

    #     return attn_output, attn_weights

    # class SurrogateLlamaAttention(SurrogateModule, LlamaAttention):
    #     def forward(
    #         self,
    #         hidden_states: torch.Tensor,
    #         position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    #         attention_mask: torch.Tensor | None = None,
    #         past_key_values=None,
    #         **kwargs,
    #     ) -> tuple[torch.Tensor, torch.Tensor]:
    #         input_shape = hidden_states.shape[:-1]
    #         hidden_shape = (*input_shape, -1, self.head_dim)

    #         query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    #         key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    #         value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    #         cos, sin = position_embeddings
    #         query_states, key_states = apply_rotary_pos_emb(
    #             query_states, key_states, cos, sin
    #         )

    #         if past_key_values is not None:
    #             key_states, value_states = past_key_values.update(
    #                 key_states, value_states, self.layer_idx
    #             )

    #         attention_interface = eager_attention_forward

    #         attn_output, attn_weights = attention_interface(
    #             self,
    #             query_states,
    #             key_states,
    #             value_states,
    #             attention_mask,
    #             dropout=0.0 if not self.training else self.attention_dropout,
    #             scaling=self.scaling,
    #             standard_backward=self.standard_backward,
    #             **kwargs,
    #         )

    #         attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    #         attn_output = self.o_proj(attn_output)
    #         return attn_output, attn_weights
