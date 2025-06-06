"""
This type stub file was generated by pyright.
"""

import torch
from torch import nn

from ... import PreTrainedModel
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPast, CausalLMOutputWithPast
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_torch_flex_attn_available,
    replace_return_docstrings,
)
from ...utils.deprecation import deprecate_kwarg
from .configuration_mllama import MllamaConfig, MllamaTextConfig, MllamaVisionConfig

"""PyTorch Mllama model."""
if is_torch_flex_attn_available(): ...
logger = ...

class MllamaPrecomputedAspectRatioEmbedding(nn.Module):
    def __init__(self, config: MllamaVisionConfig, is_gated: bool = ...) -> None: ...
    def forward(self, hidden_state: torch.Tensor, aspect_ratio_ids: torch.Tensor) -> torch.Tensor: ...

class MllamaPrecomputedPositionEmbedding(nn.Module):
    def __init__(self, config: MllamaVisionConfig) -> None: ...
    def forward(self, hidden_state: torch.Tensor, aspect_ratio_ids: torch.Tensor) -> torch.Tensor: ...

class MllamaVisionMLP(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class MllamaVisionAttention(nn.Module):
    def __init__(self, config: MllamaVisionConfig) -> None: ...
    def forward(
        self, hidden_state: torch.Tensor, attention_mask: torch.Tensor | None = ..., output_attentions: bool = ...
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

class MllamaVisionSdpaAttention(MllamaVisionAttention):
    def forward(
        self, hidden_state: torch.Tensor, attention_mask: torch.Tensor | None = ..., output_attentions: bool = ...
    ) -> torch.Tensor: ...

MLLAMA_VISION_ATTENTION_CLASSES = ...

class MllamaVisionEncoderLayer(nn.Module):
    def __init__(self, config: MllamaVisionConfig, is_gated: bool = ...) -> None: ...
    def forward(
        self, hidden_state: torch.Tensor, attention_mask: torch.Tensor | None = ..., output_attentions: bool = ...
    ):  # -> tuple[Tensor, Any] | tuple[Tensor]:
        ...

class MllamaVisionEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`MllamaEncoderLayer`].

    Args:
        config: MllamaConfig
    """
    def __init__(self, config: MllamaVisionConfig, num_layers=..., is_gated=...) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutput:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        ...

class MllamaTextRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=...) -> None:
        """
        MllamaTextRMSNorm is equivalent to T5LayerNorm
        """
        ...

    def forward(self, hidden_states): ...
    def extra_repr(self):  # -> str:
        ...

class MllamaTextCrossAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(self, config: MllamaTextConfig | None = ..., layer_idx: int | None = ...) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: torch.Tensor | None = ...,
        past_key_value: Cache | None = ...,
        attention_mask: torch.Tensor | None = ...,
        output_attentions: bool = ...,
        use_cache: bool = ...,
        cache_position: torch.LongTensor | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        """Input shape: Batch x Time x Channel"""
        ...

class MllamaTextCrossSdpaAttention(MllamaTextCrossAttention):
    """
    Mllama attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `MllamaTextCrossAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """
    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: torch.Tensor | None = ...,
        past_key_value: Cache | None = ...,
        attention_mask: torch.Tensor | None = ...,
        output_attentions: bool = ...,
        use_cache: bool = ...,
        cache_position: torch.LongTensor | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        """Input shape: Batch x Time x Channel"""
        ...

def rotate_half(x):  # -> Tensor:
    """Rotates half the hidden dims of the input."""
    ...

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=..., unsqueeze_dim=...):  # -> tuple[Any, Any]:
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    ...

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    ...

class MllamaTextSelfAttention(nn.Module):
    def __init__(self, config: MllamaTextConfig, layer_idx: int) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: torch.Tensor,
        output_attentions: bool = ...,
        use_cache: bool = ...,
        past_key_value=...,
        cache_position=...,
        **kwargs,
    ):  # -> tuple[Any, Tensor | None, Any | None]:
        ...

class MllamaTextSelfSdpaAttention(MllamaTextSelfAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: torch.Tensor,
        output_attentions: bool = ...,
        use_cache: bool = ...,
        past_key_value=...,
        cache_position=...,
        **kwargs,
    ):  # -> tuple[Any, Tensor | None, Any | None] | tuple[Any, None, Any | None]:
        ...

MLLAMA_TEXT_CROSS_ATTENTION_CLASSES = ...
MLLAMA_TEXT_ATTENTION_CLASSES = ...

class MllamaTextMLP(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, x):  # -> Any:
        ...

class MllamaSelfAttentionDecoderLayer(nn.Module):
    def __init__(self, config: MllamaTextConfig, layer_idx: int) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: torch.Tensor | None = ...,
        cross_attention_mask: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        full_text_row_masked_out_mask: tuple[torch.Tensor, torch.Tensor] | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_value: Cache | None = ...,
        output_attentions: bool | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = ...,
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        ...

class MllamaCrossAttentionDecoderLayer(torch.nn.Module):
    """Cross-attention transformer block with tanh-gated attention and feedforward."""
    def __init__(self, config: MllamaTextConfig, layer_idx: int) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: torch.Tensor,
        cross_attention_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        full_text_row_masked_out_mask: tuple[torch.Tensor, torch.Tensor],
        position_ids: torch.LongTensor | None = ...,
        past_key_value: Cache | None = ...,
        output_attentions: bool | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        position_embeddings: torch.Tensor | None = ...,
    ) -> tuple[torch.Tensor]: ...

class MllamaRotaryEmbedding(nn.Module):
    def __init__(self, config: MllamaTextConfig, device=...) -> None: ...
    @torch.no_grad()
    def forward(self, x, position_ids):  # -> tuple[Tensor | Any, Tensor | Any]:
        ...

class MllamaPreTrainedModel(PreTrainedModel):
    config_class = MllamaConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _supports_cache_class = ...
    _supports_static_cache = ...
    _supports_sdpa = ...
    _supports_quantized_cache = ...

MLLAMA_START_DOCSTRING = ...
MLLAMA_VISION_INPUTS_DOCSTRING = ...
MLLAMA_TEXT_INPUTS_DOCSTRING = ...
MLLAMA_INPUTS_DOCSTRING = ...

@add_start_docstrings("""The Mllama Vision Model which consists of two vision encoders.""", MLLAMA_START_DOCSTRING)
class MllamaVisionModel(MllamaPreTrainedModel):
    config_class = MllamaVisionConfig
    base_model_prefix = ...
    def __init__(self, config: MllamaVisionConfig) -> None: ...
    def get_input_embeddings(self):  # -> Conv2d:
        """
        This function is used to fetch the first embedding layer to activate grads on inputs.
        """
        ...

    def apply_class_embedding(self, hidden_state: torch.Tensor) -> torch.Tensor: ...
    @add_start_docstrings_to_model_forward(MLLAMA_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutput, config_class="MllamaVisionConfig")
    def forward(
        self,
        pixel_values: torch.Tensor,
        aspect_ratio_ids: torch.Tensor,
        aspect_ratio_mask: torch.Tensor,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> BaseModelOutput | tuple[torch.Tensor, ...]:
        r"""

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, MllamaVisionModel

        >>> checkpoint = "meta-llama/Llama-3.2-11B-Vision"
        >>> model = MllamaVisionModel.from_pretrained(checkpoint)
        >>> processor = AutoProcessor.from_pretrained(checkpoint)

        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = processor(images=image, return_tensors="pt")

        >>> output = model(**inputs)

        >>> print(output.last_hidden_state.shape)
        torch.Size([1, 1, 4, 1025, 7680])
        ```
        """
        ...

@add_start_docstrings(
    """The Mllama Text Model which consists of transformer with self and cross attention layers.""",
    MLLAMA_START_DOCSTRING,
)
class MllamaTextModel(MllamaPreTrainedModel):
    config_class = MllamaTextConfig
    base_model_prefix = ...
    def __init__(self, config: MllamaTextConfig) -> None: ...
    def get_input_embeddings(self):  # -> Embedding | Module:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @add_start_docstrings_to_model_forward(MLLAMA_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPast, config_class="MllamaTextConfig")
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        cross_attention_states: torch.FloatTensor | None = ...,
        cross_attention_mask: torch.Tensor | None = ...,
        full_text_row_masked_out_mask: tuple[torch.Tensor, torch.Tensor] | None = ...,
        past_key_values: Cache | list[torch.FloatTensor] | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
    ) -> tuple | BaseModelOutputWithPast:
        """

        Returns:

        Example:

        ```python
        >>> from transformers import AutoProcessor, MllamaTextModel

        >>> checkpoint = "meta-llama/Llama-3.2-11B-Vision"
        >>> model = MllamaTextModel.from_pretrained(checkpoint)
        >>> processor = AutoProcessor.from_pretrained(checkpoint)

        >>> text = "<|image|>If I had to write a haiku for this one"
        >>> inputs = processor(text=text, return_tensors="pt")

        >>> output = model(**inputs)

        >>> print(output.last_hidden_state.shape)
        torch.Size([1, 13, 4096])
        ```
        """
        ...

@add_start_docstrings("""The Mllama Text Model with a language modeling head on top.""", MLLAMA_START_DOCSTRING)
class MllamaForCausalLM(MllamaPreTrainedModel, GenerationMixin):
    config_class = MllamaTextConfig
    _supports_static_cache = ...
    base_model_prefix = ...
    _tied_weights_keys = ...
    def __init__(self, config) -> None: ...
    def get_input_embeddings(self):  # -> Embedding | Module:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def get_output_embeddings(self):  # -> Linear:
        ...
    def set_output_embeddings(self, new_embeddings):  # -> None:
        ...
    def set_decoder(self, decoder):  # -> None:
        ...
    def get_decoder(self):  # -> MllamaTextModel:
        ...
    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    @add_start_docstrings_to_model_forward(MLLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class="MllamaTextConfig")
    def forward(
        self,
        input_ids: torch.LongTensor = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        cross_attention_states: torch.LongTensor | None = ...,
        cross_attention_mask: torch.LongTensor | None = ...,
        full_text_row_masked_out_mask: tuple[torch.Tensor, torch.Tensor] | None = ...,
        past_key_values: Cache | list[torch.FloatTensor] | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
        **loss_kwargs,
    ) -> tuple | CausalLMOutputWithPast:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, MllamaForCausalLM

        >>> model = MllamaForCausalLM.from_pretrained("Llama-3.2-11B-Vision")
        >>> tokenizer = AutoTokenizer.from_pretrained("Llama-3.2-11B-Vision")

        >>> prompt = "If I had to write a haiku, it would be:"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=40, do_sample=True, temperature=0.6)
        >>> result = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        >>> print(result)
        If I had to write a haiku, it would be: "Snowflakes gently fall" - simple, yet peaceful.
        I love the idea of snowflakes gently falling, each one
        ```
        """
        ...

@add_start_docstrings(
    """The Mllama model which consists of a vision encoder and a language model.""", MLLAMA_START_DOCSTRING
)
class MllamaForConditionalGeneration(MllamaPreTrainedModel, GenerationMixin):
    _supports_quantized_cache = ...
    def __init__(self, config: MllamaConfig) -> None: ...
    def get_input_embeddings(self):  # -> Embedding | Module:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def get_output_embeddings(self):  # -> Linear:
        ...
    def set_output_embeddings(self, new_embeddings):  # -> None:
        ...
    def set_decoder(self, decoder):  # -> None:
        ...
    def get_decoder(self):  # -> MllamaTextModel:
        ...
    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    @add_start_docstrings_to_model_forward(MLLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class="MllamaConfig")
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        pixel_values: torch.FloatTensor | None = ...,
        aspect_ratio_mask: torch.Tensor | None = ...,
        aspect_ratio_ids: torch.Tensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        cross_attention_mask: torch.Tensor | None = ...,
        cross_attention_states: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: list[torch.FloatTensor] | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
    ) -> tuple | CausalLMOutputWithPast:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).


        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, MllamaForConditionalGeneration

        >>> checkpoint = "meta-llama/Llama-3.2-11B-Vision"
        >>> model = MllamaForConditionalGeneration.from_pretrained(checkpoint)
        >>> processor = AutoProcessor.from_pretrained(checkpoint)

        >>> prompt = "<|image|>If I had to write a haiku for this one"
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(text=prompt, images=image, return_tensors="pt")

        >>> # Generate
        >>> output = model.generate(**inputs, max_new_tokens=15)

        >>> prompt_len = inputs.input_ids.shape[-1]
        >>> generated_ids = output[:, prompt_len:]
        >>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        >>> print(generated_text)
        [', it would be:.\\nA stop sign in Chinatown.\\n']
        ```
        """
        ...

    def prepare_inputs_for_generation(
        self,
        input_ids=...,
        inputs_embeds=...,
        attention_mask=...,
        position_ids=...,
        pixel_values=...,
        aspect_ratio_ids=...,
        aspect_ratio_mask=...,
        cross_attention_mask=...,
        past_key_values=...,
        use_cache=...,
        cache_position=...,
        logits_to_keep=...,
        **kwargs,
    ):  # -> dict[str, Any | None]:
        ...

__all__ = [
    "MllamaForConditionalGeneration",
    "MllamaForCausalLM",
    "MllamaTextModel",
    "MllamaVisionModel",
    "MllamaPreTrainedModel",
]
