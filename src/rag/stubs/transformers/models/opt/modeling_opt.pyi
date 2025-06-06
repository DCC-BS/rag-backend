"""
This type stub file was generated by pyright.
"""

import torch
from torch import nn

from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_torch_flex_attn_available,
    replace_return_docstrings,
)
from .configuration_opt import OPTConfig

"""PyTorch OPT model."""
if is_torch_flex_attn_available(): ...
if is_flash_attn_2_available(): ...
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
_EXPECTED_OUTPUT_SHAPE = ...
_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION = ...
_SEQ_CLASS_EXPECTED_LOSS = ...
_SEQ_CLASS_EXPECTED_OUTPUT = ...

class OPTLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int) -> None: ...
    def forward(
        self,
        attention_mask: torch.LongTensor,
        past_key_values_length: int = ...,
        position_ids: torch.LongTensor | None = ...,
    ):  # -> Tensor:
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        ...

class OPTAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(self, config: OPTConfig, layer_idx: int = ..., **kwargs) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: tuple[torch.Tensor] | None = ...,
        attention_mask: torch.Tensor | None = ...,
        layer_head_mask: torch.Tensor | None = ...,
        output_attentions: bool = ...,
        position_ids: torch.Tensor | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]:
        """Input shape: Batch x Time x Channel"""
        ...

class OptFlashAttention2(OPTAttention):
    """
    OPT flash attention module. This module inherits from `OPTAttention` as the weights of the module stays untouched.
    The only required change would be on the forward pass where it needs to correctly call the public API of flash
    attention and deal with padding tokens in case the input contains any of them.
    """
    def __init__(self, *args, **kwargs) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: tuple[torch.Tensor] | None = ...,
        attention_mask: torch.Tensor | None = ...,
        layer_head_mask: torch.Tensor | None = ...,
        output_attentions: bool = ...,
        position_ids: torch.Tensor | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        """Input shape: Batch x Time x Channel"""
        ...

class OPTSdpaAttention(OPTAttention):
    """
    OPT sdpa attention module. This module inherits from `OPTAttention` as the weights of the module stays untouched.
    The only required change would be on the forward pass where it needs to correctly call the public API of sdpa
    attention and deal with padding tokens in case the input contains any of them.
    """
    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: tuple[torch.Tensor] | None = ...,
        attention_mask: torch.Tensor | None = ...,
        layer_head_mask: torch.Tensor | None = ...,
        output_attentions: bool = ...,
        position_ids: torch.Tensor | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

OPT_ATTENTION_CLASSES = ...

class OPTDecoderLayer(nn.Module):
    def __init__(self, config: OPTConfig, layer_idx: int = ...) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        layer_head_mask: torch.Tensor | None = ...,
        past_key_value: tuple[torch.Tensor] | None = ...,
        output_attentions: bool | None = ...,
        use_cache: bool | None = ...,
        position_ids: torch.LongTensor | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`, *optional*): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence..
        """
        ...

OPT_START_DOCSTRING = ...

@add_start_docstrings(
    "The bare OPT Model outputting raw hidden-states without any specific head on top.", OPT_START_DOCSTRING
)
class OPTPreTrainedModel(PreTrainedModel):
    config_class = OPTConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _supports_flash_attn_2 = ...
    _supports_sdpa = ...
    _supports_cache_class = ...
    _supports_quantized_cache = ...
    _supports_static_cache = ...

OPT_INPUTS_DOCSTRING = ...

class OPTDecoder(OPTPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`OPTDecoderLayer`]

    Args:
        config: OPTConfig
    """
    def __init__(self, config: OPTConfig) -> None: ...
    def get_input_embeddings(self):  # -> Embedding | Module:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def forward(
        self,
        input_ids: torch.LongTensor = ...,
        attention_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        past_key_values: list[torch.FloatTensor] | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        position_ids: torch.LongTensor | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> tuple | BaseModelOutputWithPast:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(num_hidden_layers, num_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.

            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
                config.n_positions - 1]`. for padding use -1.

                [What are position IDs?](../glossary#position-ids)
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
                this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
                the complete sequence length.
        """
        ...

@add_start_docstrings(
    "The bare OPT Model outputting raw hidden-states without any specific head on top.", OPT_START_DOCSTRING
)
class OPTModel(OPTPreTrainedModel):
    def __init__(self, config: OPTConfig) -> None: ...
    def get_input_embeddings(self):  # -> Embedding | Module:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def get_decoder(self):  # -> OPTDecoder:
        ...
    @add_start_docstrings_to_model_forward(OPT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        input_ids: torch.LongTensor = ...,
        attention_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        past_key_values: list[torch.FloatTensor] | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        position_ids: torch.LongTensor | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> tuple | BaseModelOutputWithPast: ...

class OPTForCausalLM(OPTPreTrainedModel, GenerationMixin):
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
    def get_decoder(self):  # -> OPTDecoder:
        ...
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = ...,
        attention_mask: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        past_key_values: list[torch.FloatTensor] | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        position_ids: torch.LongTensor | None = ...,
        cache_position: torch.Tensor | None = ...,
        **kwargs,
    ) -> tuple | CausalLMOutputWithPast:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(num_hidden_layers, num_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`. The two additional
                tensors are only required when the model is used as a decoder in a Sequence to Sequence model.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
                config.n_positions - 1]`. for padding use -1.

                [What are position IDs?](../glossary#position-ids)
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
                this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
                the complete sequence length.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, OPTForCausalLM

        >>> model = OPTForCausalLM.from_pretrained("facebook/opt-350m")
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious. I'm just a little bit of a weirdo."
        ```"""
        ...

@add_start_docstrings(
    """
    The OPT Model transformer with a sequence classification head on top (linear layer).

    [`OPTForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    OPT_START_DOCSTRING,
)
class OPTForSequenceClassification(OPTPreTrainedModel):
    def __init__(self, config: OPTConfig) -> None: ...
    @add_start_docstrings_to_model_forward(OPT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION,
        output_type=SequenceClassifierOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_SEQ_CLASS_EXPECTED_OUTPUT,
        expected_loss=_SEQ_CLASS_EXPECTED_LOSS,
    )
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        past_key_values: tuple[tuple[torch.Tensor]] | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        position_ids: torch.LongTensor | None = ...,
    ) -> tuple | SequenceClassifierOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        ...

    def get_input_embeddings(self):  # -> Embedding | Module:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...

@add_start_docstrings(
    """
    The OPT Model transformer with a span classification head on top for extractive question-answering tasks like SQuAD
    (a linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    OPT_START_DOCSTRING,
)
class OPTForQuestionAnswering(OPTPreTrainedModel):
    def __init__(self, config: OPTConfig) -> None: ...
    @add_start_docstrings_to_model_forward(OPT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=QuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        past_key_values: tuple[tuple[torch.Tensor]] | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        start_positions: torch.LongTensor | None = ...,
        end_positions: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        position_ids: torch.LongTensor | None = ...,
    ) -> tuple | QuestionAnsweringModelOutput:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, OPTForQuestionAnswering
        >>> import torch

        >>> torch.manual_seed(4)  # doctest: +IGNORE_RESULT
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

        >>> # note: we are loading a OPTForQuestionAnswering from the hub here,
        >>> # so the head will be randomly initialized, hence the predictions will be random
        >>> model = OPTForQuestionAnswering.from_pretrained("facebook/opt-350m")

        >>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"

        >>> inputs = tokenizer(question, text, return_tensors="pt")
        >>> with torch.no_grad():
        ...     outputs = model(**inputs)

        >>> answer_start_index = outputs.start_logits.argmax()
        >>> answer_end_index = outputs.end_logits.argmax()

        >>> answer_offset = len(tokenizer(question)[0])

        >>> predict_answer_tokens = inputs.input_ids[
        ...     0, answer_offset + answer_start_index : answer_offset + answer_end_index + 1
        ... ]
        >>> predicted = tokenizer.decode(predict_answer_tokens)
        >>> predicted
        ' a nice puppet'
        ```"""
        ...

    def get_input_embeddings(self):  # -> Embedding | Module:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...

__all__ = [
    "OPTForCausalLM",
    "OPTModel",
    "OPTPreTrainedModel",
    "OPTForSequenceClassification",
    "OPTForQuestionAnswering",
]
