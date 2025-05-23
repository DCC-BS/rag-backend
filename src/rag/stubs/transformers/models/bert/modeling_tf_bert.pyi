"""
This type stub file was generated by pyright.
"""

from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from ...modeling_tf_outputs import (
    TFBaseModelOutputWithPastAndCrossAttentions,
    TFBaseModelOutputWithPoolingAndCrossAttentions,
    TFCausalLMOutputWithCrossAttentions,
    TFMaskedLMOutput,
    TFMultipleChoiceModelOutput,
    TFNextSentencePredictorOutput,
    TFQuestionAnsweringModelOutput,
    TFSequenceClassifierOutput,
    TFTokenClassifierOutput,
)
from ...modeling_tf_utils import (
    TFCausalLanguageModelingLoss,
    TFMaskedLanguageModelingLoss,
    TFModelInputType,
    TFMultipleChoiceLoss,
    TFNextSentencePredictionLoss,
    TFPreTrainedModel,
    TFQuestionAnsweringLoss,
    TFSequenceClassificationLoss,
    TFTokenClassificationLoss,
    keras,
    keras_serializable,
    unpack_inputs,
)
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from .configuration_bert import BertConfig

"""TF 2.0 BERT model."""
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
_CHECKPOINT_FOR_TOKEN_CLASSIFICATION = ...
_TOKEN_CLASS_EXPECTED_OUTPUT = ...
_TOKEN_CLASS_EXPECTED_LOSS = ...
_CHECKPOINT_FOR_QA = ...
_QA_EXPECTED_OUTPUT = ...
_QA_EXPECTED_LOSS = ...
_QA_TARGET_START_INDEX = ...
_QA_TARGET_END_INDEX = ...
_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION = ...
_SEQ_CLASS_EXPECTED_OUTPUT = ...
_SEQ_CLASS_EXPECTED_LOSS = ...

class TFBertPreTrainingLoss:
    """
    Loss function suitable for BERT-like pretraining, that is, the task of pretraining a language model by combining
    NSP + MLM. .. note:: Any label of -100 will be ignored (along with the corresponding logits) in the loss
    computation.
    """
    def hf_compute_loss(self, labels: tf.Tensor, logits: tf.Tensor) -> tf.Tensor: ...

class TFBertEmbeddings(keras.layers.Layer):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, config: BertConfig, **kwargs) -> None: ...
    def build(self, input_shape=...):  # -> None:
        ...
    def call(
        self,
        input_ids: tf.Tensor = ...,
        position_ids: tf.Tensor = ...,
        token_type_ids: tf.Tensor = ...,
        inputs_embeds: tf.Tensor = ...,
        past_key_values_length=...,
        training: bool = ...,
    ) -> tf.Tensor:
        """
        Applies embedding based on inputs tensor.

        Returns:
            final_embeddings (`tf.Tensor`): output embedding tensor.
        """
        ...

class TFBertSelfAttention(keras.layers.Layer):
    def __init__(self, config: BertConfig, **kwargs) -> None: ...
    def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int) -> tf.Tensor: ...
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        head_mask: tf.Tensor,
        encoder_hidden_states: tf.Tensor,
        encoder_attention_mask: tf.Tensor,
        past_key_value: tuple[tf.Tensor],
        output_attentions: bool,
        training: bool = ...,
    ) -> tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFBertSelfOutput(keras.layers.Layer):
    def __init__(self, config: BertConfig, **kwargs) -> None: ...
    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = ...) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFBertAttention(keras.layers.Layer):
    def __init__(self, config: BertConfig, **kwargs) -> None: ...
    def prune_heads(self, heads): ...
    def call(
        self,
        input_tensor: tf.Tensor,
        attention_mask: tf.Tensor,
        head_mask: tf.Tensor,
        encoder_hidden_states: tf.Tensor,
        encoder_attention_mask: tf.Tensor,
        past_key_value: tuple[tf.Tensor],
        output_attentions: bool,
        training: bool = ...,
    ) -> tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFBertIntermediate(keras.layers.Layer):
    def __init__(self, config: BertConfig, **kwargs) -> None: ...
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFBertOutput(keras.layers.Layer):
    def __init__(self, config: BertConfig, **kwargs) -> None: ...
    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = ...) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFBertLayer(keras.layers.Layer):
    def __init__(self, config: BertConfig, **kwargs) -> None: ...
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        head_mask: tf.Tensor,
        encoder_hidden_states: tf.Tensor | None,
        encoder_attention_mask: tf.Tensor | None,
        past_key_value: tuple[tf.Tensor] | None,
        output_attentions: bool,
        training: bool = ...,
    ) -> tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFBertEncoder(keras.layers.Layer):
    def __init__(self, config: BertConfig, **kwargs) -> None: ...
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        head_mask: tf.Tensor,
        encoder_hidden_states: tf.Tensor | None,
        encoder_attention_mask: tf.Tensor | None,
        past_key_values: tuple[tuple[tf.Tensor]] | None,
        use_cache: bool | None,
        output_attentions: bool,
        output_hidden_states: bool,
        return_dict: bool,
        training: bool = ...,
    ) -> TFBaseModelOutputWithPastAndCrossAttentions | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFBertPooler(keras.layers.Layer):
    def __init__(self, config: BertConfig, **kwargs) -> None: ...
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFBertPredictionHeadTransform(keras.layers.Layer):
    def __init__(self, config: BertConfig, **kwargs) -> None: ...
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFBertLMPredictionHead(keras.layers.Layer):
    def __init__(self, config: BertConfig, input_embeddings: keras.layers.Layer, **kwargs) -> None: ...
    def build(self, input_shape=...):  # -> None:
        ...
    def get_output_embeddings(self) -> keras.layers.Layer: ...
    def set_output_embeddings(self, value: tf.Variable):  # -> None:
        ...
    def get_bias(self) -> dict[str, tf.Variable]: ...
    def set_bias(self, value: tf.Variable):  # -> None:
        ...
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor: ...

class TFBertMLMHead(keras.layers.Layer):
    def __init__(self, config: BertConfig, input_embeddings: keras.layers.Layer, **kwargs) -> None: ...
    def call(self, sequence_output: tf.Tensor) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFBertNSPHead(keras.layers.Layer):
    def __init__(self, config: BertConfig, **kwargs) -> None: ...
    def call(self, pooled_output: tf.Tensor) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

@keras_serializable
class TFBertMainLayer(keras.layers.Layer):
    config_class = BertConfig
    def __init__(self, config: BertConfig, add_pooling_layer: bool = ..., **kwargs) -> None: ...
    def get_input_embeddings(self) -> keras.layers.Layer: ...
    def set_input_embeddings(self, value: tf.Variable):  # -> None:
        ...
    @unpack_inputs
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        token_type_ids: np.ndarray | tf.Tensor | None = ...,
        position_ids: np.ndarray | tf.Tensor | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        encoder_hidden_states: np.ndarray | tf.Tensor | None = ...,
        encoder_attention_mask: np.ndarray | tf.Tensor | None = ...,
        past_key_values: tuple[tuple[np.ndarray | tf.Tensor]] | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool = ...,
    ) -> TFBaseModelOutputWithPoolingAndCrossAttentions | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFBertPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BertConfig
    base_model_prefix = ...

@dataclass
class TFBertForPreTrainingOutput(ModelOutput):
    """
    Output type of [`TFBertForPreTraining`].

    Args:
        prediction_logits (`tf.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        seq_relationship_logits (`tf.Tensor` of shape `(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
            before SoftMax).
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: tf.Tensor | None = ...
    prediction_logits: tf.Tensor = ...
    seq_relationship_logits: tf.Tensor = ...
    hidden_states: tuple[tf.Tensor] | tf.Tensor | None = ...
    attentions: tuple[tf.Tensor] | tf.Tensor | None = ...

BERT_START_DOCSTRING = ...
BERT_INPUTS_DOCSTRING = ...

@add_start_docstrings(
    "The bare Bert Model transformer outputting raw hidden-states without any specific head on top.",
    BERT_START_DOCSTRING,
)
class TFBertModel(TFBertPreTrainedModel):
    def __init__(self, config: BertConfig, add_pooling_layer: bool = ..., *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        token_type_ids: np.ndarray | tf.Tensor | None = ...,
        position_ids: np.ndarray | tf.Tensor | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        encoder_hidden_states: np.ndarray | tf.Tensor | None = ...,
        encoder_attention_mask: np.ndarray | tf.Tensor | None = ...,
        past_key_values: tuple[tuple[np.ndarray | tf.Tensor]] | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool | None = ...,
    ) -> TFBaseModelOutputWithPoolingAndCrossAttentions | tuple[tf.Tensor]:
        r"""
        encoder_hidden_states  (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

        past_key_values (`Tuple[Tuple[tf.Tensor]]` of length `config.n_layers`)
            contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*, defaults to `True`):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`). Set to `False` during training, `True` during generation
        """
        ...

    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    """
Bert Model with two heads on top as done during the pretraining:
    a `masked language modeling` head and a `next sentence prediction (classification)` head.
    """,
    BERT_START_DOCSTRING,
)
class TFBertForPreTraining(TFBertPreTrainedModel, TFBertPreTrainingLoss):
    _keys_to_ignore_on_load_unexpected = ...
    def __init__(self, config: BertConfig, *inputs, **kwargs) -> None: ...
    def get_lm_head(self) -> keras.layers.Layer: ...
    def get_prefix_bias_name(self) -> str: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TFBertForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        token_type_ids: np.ndarray | tf.Tensor | None = ...,
        position_ids: np.ndarray | tf.Tensor | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        labels: np.ndarray | tf.Tensor | None = ...,
        next_sentence_label: np.ndarray | tf.Tensor | None = ...,
        training: bool | None = ...,
    ) -> TFBertForPreTrainingOutput | tuple[tf.Tensor]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        next_sentence_label (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
            (see `input_ids` docstring) Indices should be in `[0, 1]`:

            - 0 indicates sequence B is a continuation of sequence A,
            - 1 indicates sequence B is a random sequence.
        kwargs (`Dict[str, any]`, *optional*, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.

        Return:

        Examples:

        ```python
        >>> import tensorflow as tf
        >>> from transformers import AutoTokenizer, TFBertForPreTraining

        >>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        >>> model = TFBertForPreTraining.from_pretrained("google-bert/bert-base-uncased")
        >>> input_ids = tokenizer("Hello, my dog is cute", add_special_tokens=True, return_tensors="tf")
        >>> # Batch size 1

        >>> outputs = model(input_ids)
        >>> prediction_logits, seq_relationship_logits = outputs[:2]
        ```"""
        ...

    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings("""Bert Model with a `language modeling` head on top.""", BERT_START_DOCSTRING)
class TFBertForMaskedLM(TFBertPreTrainedModel, TFMaskedLanguageModelingLoss):
    _keys_to_ignore_on_load_unexpected = ...
    def __init__(self, config: BertConfig, *inputs, **kwargs) -> None: ...
    def get_lm_head(self) -> keras.layers.Layer: ...
    def get_prefix_bias_name(self) -> str: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFMaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="'paris'",
        expected_loss=0.88,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        token_type_ids: np.ndarray | tf.Tensor | None = ...,
        position_ids: np.ndarray | tf.Tensor | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        labels: np.ndarray | tf.Tensor | None = ...,
        training: bool | None = ...,
    ) -> TFMaskedLMOutput | tuple[tf.Tensor]:
        r"""
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        ...

    def build(self, input_shape=...):  # -> None:
        ...

class TFBertLMHeadModel(TFBertPreTrainedModel, TFCausalLanguageModelingLoss):
    _keys_to_ignore_on_load_unexpected = ...
    def __init__(self, config: BertConfig, *inputs, **kwargs) -> None: ...
    def get_lm_head(self) -> keras.layers.Layer: ...
    def get_prefix_bias_name(self) -> str: ...
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=..., attention_mask=..., **model_kwargs
    ):  # -> dict[str, Any | None]:
        ...
    @unpack_inputs
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFCausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        token_type_ids: np.ndarray | tf.Tensor | None = ...,
        position_ids: np.ndarray | tf.Tensor | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        encoder_hidden_states: np.ndarray | tf.Tensor | None = ...,
        encoder_attention_mask: np.ndarray | tf.Tensor | None = ...,
        past_key_values: tuple[tuple[np.ndarray | tf.Tensor]] | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        labels: np.ndarray | tf.Tensor | None = ...,
        training: bool | None = ...,
        **kwargs,
    ) -> TFCausalLMOutputWithCrossAttentions | tuple[tf.Tensor]:
        r"""
        encoder_hidden_states  (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

        past_key_values (`Tuple[Tuple[tf.Tensor]]` of length `config.n_layers`)
            contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*, defaults to `True`):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`). Set to `False` during training, `True` during generation
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the cross entropy classification loss. Indices should be in `[0, ...,
            config.vocab_size - 1]`.
        """
        ...

    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    """Bert Model with a `next sentence prediction (classification)` head on top.""", BERT_START_DOCSTRING
)
class TFBertForNextSentencePrediction(TFBertPreTrainedModel, TFNextSentencePredictionLoss):
    _keys_to_ignore_on_load_unexpected = ...
    def __init__(self, config: BertConfig, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TFNextSentencePredictorOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        token_type_ids: np.ndarray | tf.Tensor | None = ...,
        position_ids: np.ndarray | tf.Tensor | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        next_sentence_label: np.ndarray | tf.Tensor | None = ...,
        training: bool | None = ...,
    ) -> TFNextSentencePredictorOutput | tuple[tf.Tensor]:
        r"""
        Return:

        Examples:

        ```python
        >>> import tensorflow as tf
        >>> from transformers import AutoTokenizer, TFBertForNextSentencePrediction

        >>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        >>> model = TFBertForNextSentencePrediction.from_pretrained("google-bert/bert-base-uncased")

        >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
        >>> next_sentence = "The sky is blue due to the shorter wavelength of blue light."
        >>> encoding = tokenizer(prompt, next_sentence, return_tensors="tf")

        >>> logits = model(encoding["input_ids"], token_type_ids=encoding["token_type_ids"])[0]
        >>> assert logits[0][0] < logits[0][1]  # the next sentence was random
        ```"""
        ...

    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    """
    Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """,
    BERT_START_DOCSTRING,
)
class TFBertForSequenceClassification(TFBertPreTrainedModel, TFSequenceClassificationLoss):
    _keys_to_ignore_on_load_unexpected = ...
    _keys_to_ignore_on_load_missing = ...
    def __init__(self, config: BertConfig, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION,
        output_type=TFSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_SEQ_CLASS_EXPECTED_OUTPUT,
        expected_loss=_SEQ_CLASS_EXPECTED_LOSS,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        token_type_ids: np.ndarray | tf.Tensor | None = ...,
        position_ids: np.ndarray | tf.Tensor | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        labels: np.ndarray | tf.Tensor | None = ...,
        training: bool | None = ...,
    ) -> TFSequenceClassifierOutput | tuple[tf.Tensor]:
        r"""
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        ...

    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    """
    Bert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    BERT_START_DOCSTRING,
)
class TFBertForMultipleChoice(TFBertPreTrainedModel, TFMultipleChoiceLoss):
    _keys_to_ignore_on_load_unexpected = ...
    _keys_to_ignore_on_load_missing = ...
    def __init__(self, config: BertConfig, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFMultipleChoiceModelOutput, config_class=_CONFIG_FOR_DOC
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        token_type_ids: np.ndarray | tf.Tensor | None = ...,
        position_ids: np.ndarray | tf.Tensor | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        labels: np.ndarray | tf.Tensor | None = ...,
        training: bool | None = ...,
    ) -> TFMultipleChoiceModelOutput | tuple[tf.Tensor]:
        r"""
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ..., num_choices]`
            where `num_choices` is the size of the second dimension of the input tensors. (See `input_ids` above)
        """
        ...

    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    """
    Bert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    BERT_START_DOCSTRING,
)
class TFBertForTokenClassification(TFBertPreTrainedModel, TFTokenClassificationLoss):
    _keys_to_ignore_on_load_unexpected = ...
    _keys_to_ignore_on_load_missing = ...
    def __init__(self, config: BertConfig, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_TOKEN_CLASSIFICATION,
        output_type=TFTokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_TOKEN_CLASS_EXPECTED_OUTPUT,
        expected_loss=_TOKEN_CLASS_EXPECTED_LOSS,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        token_type_ids: np.ndarray | tf.Tensor | None = ...,
        position_ids: np.ndarray | tf.Tensor | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        labels: np.ndarray | tf.Tensor | None = ...,
        training: bool | None = ...,
    ) -> TFTokenClassifierOutput | tuple[tf.Tensor]:
        r"""
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        ...

    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    """
    Bert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    BERT_START_DOCSTRING,
)
class TFBertForQuestionAnswering(TFBertPreTrainedModel, TFQuestionAnsweringLoss):
    _keys_to_ignore_on_load_unexpected = ...
    def __init__(self, config: BertConfig, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_QA,
        output_type=TFQuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
        qa_target_start_index=_QA_TARGET_START_INDEX,
        qa_target_end_index=_QA_TARGET_END_INDEX,
        expected_output=_QA_EXPECTED_OUTPUT,
        expected_loss=_QA_EXPECTED_LOSS,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = ...,
        attention_mask: np.ndarray | tf.Tensor | None = ...,
        token_type_ids: np.ndarray | tf.Tensor | None = ...,
        position_ids: np.ndarray | tf.Tensor | None = ...,
        head_mask: np.ndarray | tf.Tensor | None = ...,
        inputs_embeds: np.ndarray | tf.Tensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        start_positions: np.ndarray | tf.Tensor | None = ...,
        end_positions: np.ndarray | tf.Tensor | None = ...,
        training: bool | None = ...,
    ) -> TFQuestionAnsweringModelOutput | tuple[tf.Tensor]:
        r"""
        start_positions (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        ...

    def build(self, input_shape=...):  # -> None:
        ...

__all__ = [
    "TFBertEmbeddings",
    "TFBertForMaskedLM",
    "TFBertForMultipleChoice",
    "TFBertForNextSentencePrediction",
    "TFBertForPreTraining",
    "TFBertForQuestionAnswering",
    "TFBertForSequenceClassification",
    "TFBertForTokenClassification",
    "TFBertLMHeadModel",
    "TFBertMainLayer",
    "TFBertModel",
    "TFBertPreTrainedModel",
]
