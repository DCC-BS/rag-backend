"""
This type stub file was generated by pyright.
"""

from dataclasses import dataclass

import torch
from torch import nn

from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from .configuration_encodec import EncodecConfig

"""PyTorch EnCodec model."""
logger = ...
_CONFIG_FOR_DOC = ...

@dataclass
class EncodecOutput(ModelOutput):
    """
    Args:
        audio_codes (`torch.LongTensor`  of shape `(batch_size, nb_chunks, chunk_length)`, *optional*):
            Discret code embeddings computed using `model.encode`.
        audio_values (`torch.FlaotTensor` of shape `(batch_size, sequence_length)`, *optional*)
            Decoded audio values, obtained using the decoder part of Encodec.
    """

    audio_codes: torch.LongTensor = ...
    audio_values: torch.FloatTensor = ...

@dataclass
class EncodecEncoderOutput(ModelOutput):
    """
    Args:
        audio_codes (`torch.LongTensor`  of shape `(batch_size, nb_chunks, chunk_length)`, *optional*):
            Discret code embeddings computed using `model.encode`.
        audio_scales (`torch.Tensor` of shape `(batch_size, nb_chunks)`, *optional*):
            Scaling factor for each `audio_codes` input. This is used to unscale each chunk of audio when decoding.
    """

    audio_codes: torch.LongTensor = ...
    audio_scales: torch.FloatTensor = ...

@dataclass
class EncodecDecoderOutput(ModelOutput):
    """
    Args:
        audio_values (`torch.FloatTensor`  of shape `(batch_size, segment_length)`, *optional*):
            Decoded audio values, obtained using the decoder part of Encodec.
    """

    audio_values: torch.FloatTensor = ...

class EncodecConv1d(nn.Module):
    """Conv1d with asymmetric or causal padding and normalization."""
    def __init__(
        self, config, in_channels: int, out_channels: int, kernel_size: int, stride: int = ..., dilation: int = ...
    ) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class EncodecConvTranspose1d(nn.Module):
    """ConvTranspose1d with asymmetric or causal padding and normalization."""
    def __init__(self, config, in_channels: int, out_channels: int, kernel_size: int, stride: int = ...) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class EncodecLSTM(nn.Module):
    """
    LSTM without worrying about the hidden state, nor the layout of the data. Expects input as convolutional layout.
    """
    def __init__(self, config, dimension) -> None: ...
    def forward(self, hidden_states): ...

class EncodecResnetBlock(nn.Module):
    """
    Residual block from SEANet model as used by EnCodec.
    """
    def __init__(self, config: EncodecConfig, dim: int, dilations: list[int]) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class EncodecEncoder(nn.Module):
    """SEANet encoder as used by EnCodec."""
    def __init__(self, config: EncodecConfig) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class EncodecDecoder(nn.Module):
    """SEANet decoder as used by EnCodec."""
    def __init__(self, config: EncodecConfig) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class EncodecEuclideanCodebook(nn.Module):
    """Codebook with Euclidean distance."""
    def __init__(self, config: EncodecConfig) -> None: ...
    def quantize(self, hidden_states): ...
    def encode(self, hidden_states): ...
    def decode(self, embed_ind):  # -> Tensor:
        ...

class EncodecVectorQuantization(nn.Module):
    """
    Vector quantization implementation. Currently supports only euclidean distance.
    """
    def __init__(self, config: EncodecConfig) -> None: ...
    def encode(self, hidden_states): ...
    def decode(self, embed_ind):  # -> Tensor:
        ...

class EncodecResidualVectorQuantizer(nn.Module):
    """Residual Vector Quantizer."""
    def __init__(self, config: EncodecConfig) -> None: ...
    def get_num_quantizers_for_bandwidth(self, bandwidth: float | None = ...) -> int:
        """Return num_quantizers based on specified target bandwidth."""
        ...

    def encode(self, embeddings: torch.Tensor, bandwidth: float | None = ...) -> torch.Tensor:
        """
        Encode a given input tensor with the specified frame rate at the given bandwidth. The RVQ encode method sets
        the appropriate number of quantizers to use and returns indices for each quantizer.
        """
        ...

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode the given codes to the quantized representation."""
        ...

class EncodecPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = EncodecConfig
    base_model_prefix = ...
    main_input_name = ...

ENCODEC_START_DOCSTRING = ...
ENCODEC_INPUTS_DOCSTRING = ...

@add_start_docstrings("The EnCodec neural audio codec model.", ENCODEC_START_DOCSTRING)
class EncodecModel(EncodecPreTrainedModel):
    def __init__(self, config: EncodecConfig) -> None: ...
    def get_encoder(self):  # -> EncodecEncoder:
        ...
    def get_decoder(self):  # -> EncodecDecoder:
        ...
    def encode(
        self,
        input_values: torch.Tensor,
        padding_mask: torch.Tensor = ...,
        bandwidth: float | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor | None] | EncodecEncoderOutput:
        """
        Encodes the input audio waveform into discrete codes.

        Args:
            input_values (`torch.Tensor` of shape `(batch_size, channels, sequence_length)`):
                Float values of the input audio waveform.
            padding_mask (`torch.Tensor` of shape `(batch_size, channels, sequence_length)`):
                Padding mask used to pad the `input_values`.
            bandwidth (`float`, *optional*):
                The target bandwidth. Must be one of `config.target_bandwidths`. If `None`, uses the smallest possible
                bandwidth. bandwidth is represented as a thousandth of what it is, e.g. 6kbps bandwidth is represented
                as bandwidth == 6.0

        Returns:
            A list of frames containing the discrete encoded codes for the input audio waveform, along with rescaling
            factors for each chunk when `normalize` is True. Each frames is a tuple `(codebook, scale)`, with
            `codebook` of shape `[batch_size, num_codebooks, frames]`.
        """
        ...

    def decode(
        self,
        audio_codes: torch.Tensor,
        audio_scales: torch.Tensor,
        padding_mask: torch.Tensor | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor] | EncodecDecoderOutput:
        """
        Decodes the given frames into an output audio waveform.

        Note that the output might be a bit bigger than the input. In that case, any extra steps at the end can be
        trimmed.

        Args:
            audio_codes (`torch.LongTensor`  of shape `(batch_size, nb_chunks, chunk_length)`, *optional*):
                Discret code embeddings computed using `model.encode`.
            audio_scales (`torch.Tensor` of shape `(batch_size, nb_chunks)`, *optional*):
                Scaling factor for each `audio_codes` input.
            padding_mask (`torch.Tensor` of shape `(batch_size, channels, sequence_length)`):
                Padding mask used to pad the `input_values`.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        """
        ...

    @add_start_docstrings_to_model_forward(ENCODEC_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=EncodecOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_values: torch.Tensor,
        padding_mask: torch.Tensor | None = ...,
        bandwidth: float | None = ...,
        audio_codes: torch.Tensor | None = ...,
        audio_scales: torch.Tensor | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.Tensor, torch.Tensor] | EncodecOutput:
        r"""
        Returns:

        Examples:

        ```python
        >>> from datasets import load_dataset
        >>> from transformers import AutoProcessor, EncodecModel

        >>> dataset = load_dataset("hf-internal-testing/ashraq-esc50-1-dog-example")
        >>> audio_sample = dataset["train"]["audio"][0]["array"]

        >>> model_id = "facebook/encodec_24khz"
        >>> model = EncodecModel.from_pretrained(model_id)
        >>> processor = AutoProcessor.from_pretrained(model_id)

        >>> inputs = processor(raw_audio=audio_sample, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> audio_codes = outputs.audio_codes
        >>> audio_values = outputs.audio_values
        ```"""
        ...

__all__ = ["EncodecModel", "EncodecPreTrainedModel"]
