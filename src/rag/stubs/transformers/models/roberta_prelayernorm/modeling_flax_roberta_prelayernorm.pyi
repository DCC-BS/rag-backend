"""
This type stub file was generated by pyright.
"""

from collections.abc import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict

from ...modeling_flax_utils import FlaxPreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward
from .configuration_roberta_prelayernorm import RobertaPreLayerNormConfig

"""Flax RoBERTa-PreLayerNorm model."""
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
remat = ...

def create_position_ids_from_input_ids(input_ids, padding_idx):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        input_ids: jnp.ndarray
        padding_idx: int

    Returns: jnp.ndarray
    """
    ...

ROBERTA_PRELAYERNORM_START_DOCSTRING = ...
ROBERTA_PRELAYERNORM_INPUTS_DOCSTRING = ...

class FlaxRobertaPreLayerNormEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    config: RobertaPreLayerNormConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, input_ids, token_type_ids, position_ids, attention_mask, deterministic: bool = ...): ...

class FlaxRobertaPreLayerNormSelfAttention(nn.Module):
    config: RobertaPreLayerNormConfig
    causal: bool = ...
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        hidden_states,
        attention_mask,
        layer_head_mask,
        key_value_states: jnp.ndarray | None = ...,
        init_cache: bool = ...,
        deterministic=...,
        output_attentions: bool = ...,
    ):  # -> tuple[Any, Any] | tuple[Any]:
        ...

class FlaxRobertaPreLayerNormSelfOutput(nn.Module):
    config: RobertaPreLayerNormConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states, input_tensor, deterministic: bool = ...): ...

class FlaxRobertaPreLayerNormAttention(nn.Module):
    config: RobertaPreLayerNormConfig
    causal: bool = ...
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        hidden_states,
        attention_mask,
        layer_head_mask,
        key_value_states=...,
        init_cache=...,
        deterministic=...,
        output_attentions: bool = ...,
    ):  # -> tuple[Any, Any] | tuple[Any]:
        ...

class FlaxRobertaPreLayerNormIntermediate(nn.Module):
    config: RobertaPreLayerNormConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states): ...

class FlaxRobertaPreLayerNormOutput(nn.Module):
    config: RobertaPreLayerNormConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states, attention_output, deterministic: bool = ...): ...

class FlaxRobertaPreLayerNormLayer(nn.Module):
    config: RobertaPreLayerNormConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        hidden_states,
        attention_mask,
        layer_head_mask,
        encoder_hidden_states: jnp.ndarray | None = ...,
        encoder_attention_mask: jnp.ndarray | None = ...,
        init_cache: bool = ...,
        deterministic: bool = ...,
        output_attentions: bool = ...,
    ):  # -> tuple[Any, Any, Any] | tuple[Any, Any] | tuple[Any]:
        ...

class FlaxRobertaPreLayerNormLayerCollection(nn.Module):
    config: RobertaPreLayerNormConfig
    dtype: jnp.dtype = ...
    gradient_checkpointing: bool = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        hidden_states,
        attention_mask,
        head_mask,
        encoder_hidden_states: jnp.ndarray | None = ...,
        encoder_attention_mask: jnp.ndarray | None = ...,
        init_cache: bool = ...,
        deterministic: bool = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ):  # -> tuple[Any | tuple[Any, ...] | tuple[()], ...] | FlaxBaseModelOutputWithPastAndCrossAttentions:
        ...

class FlaxRobertaPreLayerNormEncoder(nn.Module):
    config: RobertaPreLayerNormConfig
    dtype: jnp.dtype = ...
    gradient_checkpointing: bool = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        hidden_states,
        attention_mask,
        head_mask,
        encoder_hidden_states: jnp.ndarray | None = ...,
        encoder_attention_mask: jnp.ndarray | None = ...,
        init_cache: bool = ...,
        deterministic: bool = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ):  # -> tuple[Any | tuple[Any, ...] | tuple[()], ...] | FlaxBaseModelOutputWithPastAndCrossAttentions:
        ...

class FlaxRobertaPreLayerNormPooler(nn.Module):
    config: RobertaPreLayerNormConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states): ...

class FlaxRobertaPreLayerNormLMHead(nn.Module):
    config: RobertaPreLayerNormConfig
    dtype: jnp.dtype = ...
    bias_init: Callable[..., np.ndarray] = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states, shared_embedding=...): ...

class FlaxRobertaPreLayerNormClassificationHead(nn.Module):
    config: RobertaPreLayerNormConfig
    dtype: jnp.dtype = ...
    def setup(self):  # -> None:
        ...
    def __call__(self, hidden_states, deterministic=...): ...

class FlaxRobertaPreLayerNormPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = RobertaPreLayerNormConfig
    base_model_prefix = ...
    module_class: nn.Module = ...
    def __init__(
        self,
        config: RobertaPreLayerNormConfig,
        input_shape: tuple = ...,
        seed: int = ...,
        dtype: jnp.dtype = ...,
        _do_init: bool = ...,
        gradient_checkpointing: bool = ...,
        **kwargs,
    ) -> None: ...
    def enable_gradient_checkpointing(self):  # -> None:
        ...
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: tuple, params: FrozenDict = ...) -> FrozenDict: ...
    def init_cache(self, batch_size, max_length):
        r"""
        Args:
            batch_size (`int`):
                batch_size used for fast auto-regressive decoding. Defines the batch size of the initialized cache.
            max_length (`int`):
                maximum possible length for auto-regressive decoding. Defines the sequence length of the initialized
                cache.
        """
        ...

    @add_start_docstrings_to_model_forward(ROBERTA_PRELAYERNORM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def __call__(
        self,
        input_ids,
        attention_mask=...,
        token_type_ids=...,
        position_ids=...,
        head_mask=...,
        encoder_hidden_states=...,
        encoder_attention_mask=...,
        params: dict = ...,
        dropout_rng: jax.random.PRNGKey = ...,
        train: bool = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        past_key_values: dict = ...,
    ): ...

class FlaxRobertaPreLayerNormModule(nn.Module):
    config: RobertaPreLayerNormConfig
    dtype: jnp.dtype = ...
    add_pooling_layer: bool = ...
    gradient_checkpointing: bool = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids: jnp.ndarray | None = ...,
        position_ids: jnp.ndarray | None = ...,
        head_mask: jnp.ndarray | None = ...,
        encoder_hidden_states: jnp.ndarray | None = ...,
        encoder_attention_mask: jnp.ndarray | None = ...,
        init_cache: bool = ...,
        deterministic: bool = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ):  # -> tuple[Any, *tuple[Any | tuple[Any, ...] | tuple[()], ...]] | Any | tuple[Any, Any, *tuple[Any | tuple[Any, ...] | tuple[()], ...]] | FlaxBaseModelOutputWithPoolingAndCrossAttentions:
        ...

@add_start_docstrings(
    "The bare RoBERTa-PreLayerNorm Model transformer outputting raw hidden-states without any specific head on top.",
    ROBERTA_PRELAYERNORM_START_DOCSTRING,
)
class FlaxRobertaPreLayerNormModel(FlaxRobertaPreLayerNormPreTrainedModel):
    module_class = ...

class FlaxRobertaPreLayerNormForMaskedLMModule(nn.Module):
    config: RobertaPreLayerNormConfig
    dtype: jnp.dtype = ...
    gradient_checkpointing: bool = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        head_mask,
        deterministic: bool = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ):  # -> tuple[Any, *tuple[Any | tuple[Any, ...] | tuple[()], ...]] | Any | tuple[Any, Any, *tuple[Any | tuple[Any, ...] | tuple[()], ...]] | FlaxMaskedLMOutput:
        ...

@add_start_docstrings(
    """RoBERTa-PreLayerNorm Model with a `language modeling` head on top.""", ROBERTA_PRELAYERNORM_START_DOCSTRING
)
class FlaxRobertaPreLayerNormForMaskedLM(FlaxRobertaPreLayerNormPreTrainedModel):
    module_class = ...

class FlaxRobertaPreLayerNormForSequenceClassificationModule(nn.Module):
    config: RobertaPreLayerNormConfig
    dtype: jnp.dtype = ...
    gradient_checkpointing: bool = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        head_mask,
        deterministic: bool = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ):  # -> tuple[Any, *tuple[Any | tuple[Any, ...] | tuple[()], ...]] | Any | tuple[Any, Any, *tuple[Any | tuple[Any, ...] | tuple[()], ...]] | FlaxSequenceClassifierOutput:
        ...

@add_start_docstrings(
    """
    RobertaPreLayerNorm Model transformer with a sequence classification/regression head on top (a linear layer on top
    of the pooled output) e.g. for GLUE tasks.
    """,
    ROBERTA_PRELAYERNORM_START_DOCSTRING,
)
class FlaxRobertaPreLayerNormForSequenceClassification(FlaxRobertaPreLayerNormPreTrainedModel):
    module_class = ...

class FlaxRobertaPreLayerNormForMultipleChoiceModule(nn.Module):
    config: RobertaPreLayerNormConfig
    dtype: jnp.dtype = ...
    gradient_checkpointing: bool = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        head_mask,
        deterministic: bool = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ):  # -> tuple[Any, *tuple[Any | tuple[Any, ...] | tuple[()], ...]] | Any | FlaxMultipleChoiceModelOutput:
        ...

@add_start_docstrings(
    """
    RobertaPreLayerNorm Model with a multiple choice classification head on top (a linear layer on top of the pooled
    output and a softmax) e.g. for RocStories/SWAG tasks.
    """,
    ROBERTA_PRELAYERNORM_START_DOCSTRING,
)
class FlaxRobertaPreLayerNormForMultipleChoice(FlaxRobertaPreLayerNormPreTrainedModel):
    module_class = ...

class FlaxRobertaPreLayerNormForTokenClassificationModule(nn.Module):
    config: RobertaPreLayerNormConfig
    dtype: jnp.dtype = ...
    gradient_checkpointing: bool = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        head_mask,
        deterministic: bool = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ):  # -> tuple[Any, *tuple[Any | tuple[Any, ...] | tuple[()], ...]] | Any | tuple[Any, Any, *tuple[Any | tuple[Any, ...] | tuple[()], ...]] | FlaxTokenClassifierOutput:
        ...

@add_start_docstrings(
    """
    RobertaPreLayerNorm Model with a token classification head on top (a linear layer on top of the hidden-states
    output) e.g. for Named-Entity-Recognition (NER) tasks.
    """,
    ROBERTA_PRELAYERNORM_START_DOCSTRING,
)
class FlaxRobertaPreLayerNormForTokenClassification(FlaxRobertaPreLayerNormPreTrainedModel):
    module_class = ...

class FlaxRobertaPreLayerNormForQuestionAnsweringModule(nn.Module):
    config: RobertaPreLayerNormConfig
    dtype: jnp.dtype = ...
    gradient_checkpointing: bool = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        head_mask,
        deterministic: bool = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ):  # -> tuple[Any, Any, *tuple[Any | tuple[Any, ...] | tuple[()], ...]] | Any | tuple[Any, Any, Any, *tuple[Any | tuple[Any, ...] | tuple[()], ...]] | FlaxQuestionAnsweringModelOutput:
        ...

@add_start_docstrings(
    """
    RobertaPreLayerNorm Model with a span classification head on top for extractive question-answering tasks like SQuAD
    (a linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    ROBERTA_PRELAYERNORM_START_DOCSTRING,
)
class FlaxRobertaPreLayerNormForQuestionAnswering(FlaxRobertaPreLayerNormPreTrainedModel):
    module_class = ...

class FlaxRobertaPreLayerNormForCausalLMModule(nn.Module):
    config: RobertaPreLayerNormConfig
    dtype: jnp.dtype = ...
    gradient_checkpointing: bool = ...
    def setup(self):  # -> None:
        ...
    def __call__(
        self,
        input_ids,
        attention_mask,
        position_ids,
        token_type_ids: jnp.ndarray | None = ...,
        head_mask: jnp.ndarray | None = ...,
        encoder_hidden_states: jnp.ndarray | None = ...,
        encoder_attention_mask: jnp.ndarray | None = ...,
        init_cache: bool = ...,
        deterministic: bool = ...,
        output_attentions: bool = ...,
        output_hidden_states: bool = ...,
        return_dict: bool = ...,
    ):  # -> tuple[Any, *tuple[Any | tuple[Any, ...] | tuple[()], ...]] | Any | tuple[Any, Any, *tuple[Any | tuple[Any, ...] | tuple[()], ...]] | FlaxCausalLMOutputWithCrossAttentions:
        ...

@add_start_docstrings(
    """
    RobertaPreLayerNorm Model with a language modeling head on top (a linear layer on top of the hidden-states output)
    e.g for autoregressive tasks.
    """,
    ROBERTA_PRELAYERNORM_START_DOCSTRING,
)
class FlaxRobertaPreLayerNormForCausalLM(FlaxRobertaPreLayerNormPreTrainedModel):
    module_class = ...
    def prepare_inputs_for_generation(
        self, input_ids, max_length, attention_mask: jax.Array | None = ...
    ):  # -> dict[str, Any]:
        ...
    def update_inputs_for_generation(self, model_outputs, model_kwargs): ...

__all__ = [
    "FlaxRobertaPreLayerNormForCausalLM",
    "FlaxRobertaPreLayerNormForMaskedLM",
    "FlaxRobertaPreLayerNormForMultipleChoice",
    "FlaxRobertaPreLayerNormForQuestionAnswering",
    "FlaxRobertaPreLayerNormForSequenceClassification",
    "FlaxRobertaPreLayerNormForTokenClassification",
    "FlaxRobertaPreLayerNormModel",
    "FlaxRobertaPreLayerNormPreTrainedModel",
]
