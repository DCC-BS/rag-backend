"""
This type stub file was generated by pyright.
"""

from dataclasses import dataclass

import tensorflow as tf

from ...modeling_tf_outputs import TFImageClassifierOutputWithNoAttention
from ...modeling_tf_utils import (
    TFModelInputType,
    TFPreTrainedModel,
    TFSequenceClassificationLoss,
    keras,
    keras_serializable,
    unpack_inputs,
)
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from .configuration_cvt import CvtConfig

"""TF 2.0 Cvt model."""
logger = ...
_CONFIG_FOR_DOC = ...

@dataclass
class TFBaseModelOutputWithCLSToken(ModelOutput):
    """
    Base class for model's outputs.

    Args:
        last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        cls_token_value (`tf.Tensor` of shape `(batch_size, 1, hidden_size)`):
            Classification token at the output of the last layer of the model.
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer plus
            the initial embedding outputs.
    """

    last_hidden_state: tf.Tensor = ...
    cls_token_value: tf.Tensor = ...
    hidden_states: tuple[tf.Tensor, ...] | None = ...

class TFCvtDropPath(keras.layers.Layer):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    References:
        (1) github.com:rwightman/pytorch-image-models
    """
    def __init__(self, drop_prob: float, **kwargs) -> None: ...
    def call(self, x: tf.Tensor, training=...): ...

class TFCvtEmbeddings(keras.layers.Layer):
    """Construct the Convolutional Token Embeddings."""
    def __init__(
        self,
        config: CvtConfig,
        patch_size: int,
        num_channels: int,
        embed_dim: int,
        stride: int,
        padding: int,
        dropout_rate: float,
        **kwargs,
    ) -> None: ...
    def call(self, pixel_values: tf.Tensor, training: bool = ...) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFCvtConvEmbeddings(keras.layers.Layer):
    """Image to Convolution Embeddings. This convolutional operation aims to model local spatial contexts."""
    def __init__(
        self, config: CvtConfig, patch_size: int, num_channels: int, embed_dim: int, stride: int, padding: int, **kwargs
    ) -> None: ...
    def call(self, pixel_values: tf.Tensor) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFCvtSelfAttentionConvProjection(keras.layers.Layer):
    """Convolutional projection layer."""
    def __init__(
        self, config: CvtConfig, embed_dim: int, kernel_size: int, stride: int, padding: int, **kwargs
    ) -> None: ...
    def call(self, hidden_state: tf.Tensor, training: bool = ...) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFCvtSelfAttentionLinearProjection(keras.layers.Layer):
    """Linear projection layer used to flatten tokens into 1D."""
    def call(self, hidden_state: tf.Tensor) -> tf.Tensor: ...

class TFCvtSelfAttentionProjection(keras.layers.Layer):
    """Convolutional Projection for Attention."""
    def __init__(
        self,
        config: CvtConfig,
        embed_dim: int,
        kernel_size: int,
        stride: int,
        padding: int,
        projection_method: str = ...,
        **kwargs,
    ) -> None: ...
    def call(self, hidden_state: tf.Tensor, training: bool = ...) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFCvtSelfAttention(keras.layers.Layer):
    """
    Self-attention layer. A depth-wise separable convolution operation (Convolutional Projection), is applied for
    query, key, and value embeddings.
    """
    def __init__(
        self,
        config: CvtConfig,
        num_heads: int,
        embed_dim: int,
        kernel_size: int,
        stride_q: int,
        stride_kv: int,
        padding_q: int,
        padding_kv: int,
        qkv_projection_method: str,
        qkv_bias: bool,
        attention_drop_rate: float,
        with_cls_token: bool = ...,
        **kwargs,
    ) -> None: ...
    def rearrange_for_multi_head_attention(self, hidden_state: tf.Tensor) -> tf.Tensor: ...
    def call(self, hidden_state: tf.Tensor, height: int, width: int, training: bool = ...) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFCvtSelfOutput(keras.layers.Layer):
    """Output of the Attention layer ."""
    def __init__(self, config: CvtConfig, embed_dim: int, drop_rate: float, **kwargs) -> None: ...
    def call(self, hidden_state: tf.Tensor, training: bool = ...) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFCvtAttention(keras.layers.Layer):
    """Attention layer. First chunk of the convolutional transformer block."""
    def __init__(
        self,
        config: CvtConfig,
        num_heads: int,
        embed_dim: int,
        kernel_size: int,
        stride_q: int,
        stride_kv: int,
        padding_q: int,
        padding_kv: int,
        qkv_projection_method: str,
        qkv_bias: bool,
        attention_drop_rate: float,
        drop_rate: float,
        with_cls_token: bool = ...,
        **kwargs,
    ) -> None: ...
    def prune_heads(self, heads): ...
    def call(self, hidden_state: tf.Tensor, height: int, width: int, training: bool = ...): ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFCvtIntermediate(keras.layers.Layer):
    """Intermediate dense layer. Second chunk of the convolutional transformer block."""
    def __init__(self, config: CvtConfig, embed_dim: int, mlp_ratio: int, **kwargs) -> None: ...
    def call(self, hidden_state: tf.Tensor) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFCvtOutput(keras.layers.Layer):
    """
    Output of the Convolutional Transformer Block (last chunk). It consists of a MLP and a residual connection.
    """
    def __init__(self, config: CvtConfig, embed_dim: int, mlp_ratio: int, drop_rate: int, **kwargs) -> None: ...
    def call(self, hidden_state: tf.Tensor, input_tensor: tf.Tensor, training: bool = ...) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFCvtLayer(keras.layers.Layer):
    """
    Convolutional Transformer Block composed by attention layers, normalization and multi-layer perceptrons (mlps). It
    consists of 3 chunks : an attention layer, an intermediate dense layer and an output layer. This corresponds to the
    `Block` class in the original implementation.
    """
    def __init__(
        self,
        config: CvtConfig,
        num_heads: int,
        embed_dim: int,
        kernel_size: int,
        stride_q: int,
        stride_kv: int,
        padding_q: int,
        padding_kv: int,
        qkv_projection_method: str,
        qkv_bias: bool,
        attention_drop_rate: float,
        drop_rate: float,
        mlp_ratio: float,
        drop_path_rate: float,
        with_cls_token: bool = ...,
        **kwargs,
    ) -> None: ...
    def call(self, hidden_state: tf.Tensor, height: int, width: int, training: bool = ...) -> tf.Tensor: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFCvtStage(keras.layers.Layer):
    """
    Cvt stage (encoder block). Each stage has 2 parts :
    - (1) A Convolutional Token Embedding layer
    - (2) A Convolutional Transformer Block (layer).
    The classification token is added only in the last stage.

    Args:
        config ([`CvtConfig`]): Model configuration class.
        stage (`int`): Stage number.
    """
    def __init__(self, config: CvtConfig, stage: int, **kwargs) -> None: ...
    def call(self, hidden_state: tf.Tensor, training: bool = ...):  # -> tuple[Any, Any | None]:
        ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFCvtEncoder(keras.layers.Layer):
    """
    Convolutional Vision Transformer encoder. CVT has 3 stages of encoder blocks with their respective number of layers
    (depth) being 1, 2 and 10.

    Args:
        config ([`CvtConfig`]): Model configuration class.
    """

    config_class = CvtConfig
    def __init__(self, config: CvtConfig, **kwargs) -> None: ...
    def call(
        self,
        pixel_values: TFModelInputType,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool | None = ...,
    ) -> TFBaseModelOutputWithCLSToken | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

@keras_serializable
class TFCvtMainLayer(keras.layers.Layer):
    """Construct the Cvt model."""

    config_class = CvtConfig
    def __init__(self, config: CvtConfig, **kwargs) -> None: ...
    @unpack_inputs
    def call(
        self,
        pixel_values: TFModelInputType | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool | None = ...,
    ) -> TFBaseModelOutputWithCLSToken | tuple[tf.Tensor]: ...
    def build(self, input_shape=...):  # -> None:
        ...

class TFCvtPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = CvtConfig
    base_model_prefix = ...
    main_input_name = ...

TFCVT_START_DOCSTRING = ...
TFCVT_INPUTS_DOCSTRING = ...

@add_start_docstrings(
    "The bare Cvt Model transformer outputting raw hidden-states without any specific head on top.",
    TFCVT_START_DOCSTRING,
)
class TFCvtModel(TFCvtPreTrainedModel):
    def __init__(self, config: CvtConfig, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(TFCVT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFBaseModelOutputWithCLSToken, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        pixel_values: tf.Tensor | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool | None = ...,
    ) -> TFBaseModelOutputWithCLSToken | tuple[tf.Tensor]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, TFCvtModel
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("microsoft/cvt-13")
        >>> model = TFCvtModel.from_pretrained("microsoft/cvt-13")

        >>> inputs = image_processor(images=image, return_tensors="tf")
        >>> outputs = model(**inputs)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        ...

    def build(self, input_shape=...):  # -> None:
        ...

@add_start_docstrings(
    """
    Cvt Model transformer with an image classification head on top (a linear layer on top of the final hidden state of
    the [CLS] token) e.g. for ImageNet.
    """,
    TFCVT_START_DOCSTRING,
)
class TFCvtForImageClassification(TFCvtPreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config: CvtConfig, *inputs, **kwargs) -> None: ...
    @unpack_inputs
    @add_start_docstrings_to_model_forward(TFCVT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFImageClassifierOutputWithNoAttention, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        pixel_values: tf.Tensor | None = ...,
        labels: tf.Tensor | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        training: bool | None = ...,
    ) -> TFImageClassifierOutputWithNoAttention | tuple[tf.Tensor]:
        r"""
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, TFCvtForImageClassification
        >>> import tensorflow as tf
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("microsoft/cvt-13")
        >>> model = TFCvtForImageClassification.from_pretrained("microsoft/cvt-13")

        >>> inputs = image_processor(images=image, return_tensors="tf")
        >>> outputs = model(**inputs)
        >>> logits = outputs.logits
        >>> # model predicts one of the 1000 ImageNet classes
        >>> predicted_class_idx = tf.math.argmax(logits, axis=-1)[0]
        >>> print("Predicted class:", model.config.id2label[int(predicted_class_idx)])
        ```"""
        ...

    def build(self, input_shape=...):  # -> None:
        ...

__all__ = ["TFCvtForImageClassification", "TFCvtModel", "TFCvtPreTrainedModel"]
