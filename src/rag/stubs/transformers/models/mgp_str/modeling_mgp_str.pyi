"""
This type stub file was generated by pyright.
"""

from dataclasses import dataclass

import torch
from torch import nn

from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from .configuration_mgp_str import MgpstrConfig

"""PyTorch MGP-STR model."""
logger = ...
_CONFIG_FOR_DOC = ...
_TOKENIZER_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...

def drop_path(input: torch.Tensor, drop_prob: float = ..., training: bool = ...) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    ...

class MgpstrDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob: float | None = ...) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
    def extra_repr(self) -> str: ...

@dataclass
class MgpstrModelOutput(ModelOutput):
    """
    Base class for vision model's outputs that also contains image embeddings of the pooling of the last hidden states.

    Args:
        logits (`tuple(torch.FloatTensor)` of shape `(batch_size, config.num_character_labels)`):
            Tuple of `torch.FloatTensor` (one for the output of character of shape `(batch_size,
            config.max_token_length, config.num_character_labels)`, + one for the output of bpe of shape `(batch_size,
            config.max_token_length, config.num_bpe_labels)`, + one for the output of wordpiece of shape `(batch_size,
            config.max_token_length, config.num_wordpiece_labels)`) .

            Classification scores (before SoftMax) of character, bpe and wordpiece.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, config.max_token_length,
            sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        a3_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_a3_attentions=True` is passed or when `config.output_a3_attentions=True`):
            Tuple of `torch.FloatTensor` (one for the attention of character, + one for the attention of bpe`, + one
            for the attention of wordpiece) of shape `(batch_size, config.max_token_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    logits: tuple[torch.FloatTensor] = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...
    a3_attentions: tuple[torch.FloatTensor] | None = ...

class MgpstrEmbeddings(nn.Module):
    """2D Image to Patch Embedding"""
    def __init__(self, config: MgpstrConfig) -> None: ...
    def forward(self, pixel_values):  # -> Any:
        ...

class MgpstrMlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""
    def __init__(self, config: MgpstrConfig, hidden_features) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class MgpstrAttention(nn.Module):
    def __init__(self, config: MgpstrConfig) -> None: ...
    def forward(self, hidden_states):  # -> tuple[Any, Any]:
        ...

class MgpstrLayer(nn.Module):
    def __init__(self, config: MgpstrConfig, drop_path=...) -> None: ...
    def forward(self, hidden_states):  # -> tuple[Any, Any]:
        ...

class MgpstrEncoder(nn.Module):
    def __init__(self, config: MgpstrConfig) -> None: ...
    def forward(
        self, hidden_states, output_attentions=..., output_hidden_states=..., return_dict=...
    ):  # -> tuple[Any | tuple[Any, ...] | tuple[()], ...] | BaseModelOutput:
        ...

class MgpstrA3Module(nn.Module):
    def __init__(self, config: MgpstrConfig) -> None: ...
    def forward(self, hidden_states):  # -> tuple[Any, Tensor]:
        ...

class MgpstrPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = MgpstrConfig
    base_model_prefix = ...
    _no_split_modules = ...

MGP_STR_START_DOCSTRING = ...
MGP_STR_INPUTS_DOCSTRING = ...

@add_start_docstrings(
    "The bare MGP-STR Model transformer outputting raw hidden-states without any specific head on top.",
    MGP_STR_START_DOCSTRING,
)
class MgpstrModel(MgpstrPreTrainedModel):
    def __init__(self, config: MgpstrConfig) -> None: ...
    def get_input_embeddings(self) -> nn.Module: ...
    @add_start_docstrings_to_model_forward(MGP_STR_INPUTS_DOCSTRING)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.FloatTensor] | BaseModelOutput: ...

@add_start_docstrings(
    """
    MGP-STR Model transformer with three classification heads on top (three A^3 modules and three linear layer on top
    of the transformer encoder output) for scene text recognition (STR) .
    """,
    MGP_STR_START_DOCSTRING,
)
class MgpstrForSceneTextRecognition(MgpstrPreTrainedModel):
    config_class = MgpstrConfig
    main_input_name = ...
    def __init__(self, config: MgpstrConfig) -> None: ...
    @add_start_docstrings_to_model_forward(MGP_STR_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=MgpstrModelOutput, config_class=MgpstrConfig)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: bool | None = ...,
        output_a3_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.FloatTensor] | MgpstrModelOutput:
        r"""
        output_a3_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of a3 modules. See `a3_attentions` under returned tensors
            for more detail.

        Returns:

        Example:

        ```python
        >>> from transformers import (
        ...     MgpstrProcessor,
        ...     MgpstrForSceneTextRecognition,
        ... )
        >>> import requests
        >>> from PIL import Image

        >>> # load image from the IIIT-5k dataset
        >>> url = "https://i.postimg.cc/ZKwLg2Gw/367-14.png"
        >>> image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

        >>> processor = MgpstrProcessor.from_pretrained("alibaba-damo/mgp-str-base")
        >>> pixel_values = processor(images=image, return_tensors="pt").pixel_values

        >>> model = MgpstrForSceneTextRecognition.from_pretrained("alibaba-damo/mgp-str-base")

        >>> # inference
        >>> outputs = model(pixel_values)
        >>> out_strs = processor.batch_decode(outputs.logits)
        >>> out_strs["generated_text"]
        '["ticket"]'
        ```"""
        ...

__all__ = ["MgpstrModel", "MgpstrPreTrainedModel", "MgpstrForSceneTextRecognition"]
