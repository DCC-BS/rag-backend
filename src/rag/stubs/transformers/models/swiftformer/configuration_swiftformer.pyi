"""
This type stub file was generated by pyright.
"""

from collections.abc import Mapping

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig

"""SwiftFormer model configuration"""
logger = ...

class SwiftFormerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`SwiftFormerModel`]. It is used to instantiate an
    SwiftFormer model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the SwiftFormer
    [MBZUAI/swiftformer-xs](https://huggingface.co/MBZUAI/swiftformer-xs) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels
        depths (`List[int]`, *optional*, defaults to `[3, 3, 6, 4]`):
            Depth of each stage
        embed_dims (`List[int]`, *optional*, defaults to `[48, 56, 112, 220]`):
            The embedding dimension at each stage
        mlp_ratio (`int`, *optional*, defaults to 4):
            Ratio of size of the hidden dimensionality of an MLP to the dimensionality of its input.
        downsamples (`List[bool]`, *optional*, defaults to `[True, True, True, True]`):
            Whether or not to downsample inputs between two stages.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (string). `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` are supported.
        down_patch_size (`int`, *optional*, defaults to 3):
            The size of patches in downsampling layers.
        down_stride (`int`, *optional*, defaults to 2):
            The stride of convolution kernels in downsampling layers.
        down_pad (`int`, *optional*, defaults to 1):
            Padding in downsampling layers.
        drop_path_rate (`float`, *optional*, defaults to 0.0):
            Rate at which to increase dropout probability in DropPath.
        drop_mlp_rate (`float`, *optional*, defaults to 0.0):
            Dropout rate for the MLP component of SwiftFormer.
        drop_conv_encoder_rate (`float`, *optional*, defaults to 0.0):
            Dropout rate for the ConvEncoder component of SwiftFormer.
        use_layer_scale (`bool`, *optional*, defaults to `True`):
            Whether to scale outputs from token mixers.
        layer_scale_init_value (`float`, *optional*, defaults to 1e-05):
            Factor by which outputs from token mixers are scaled.
        batch_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the batch normalization layers.


    Example:

    ```python
    >>> from transformers import SwiftFormerConfig, SwiftFormerModel

    >>> # Initializing a SwiftFormer swiftformer-base-patch16-224 style configuration
    >>> configuration = SwiftFormerConfig()

    >>> # Initializing a model (with random weights) from the swiftformer-base-patch16-224 style configuration
    >>> model = SwiftFormerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = ...
    def __init__(
        self,
        image_size=...,
        num_channels=...,
        depths=...,
        embed_dims=...,
        mlp_ratio=...,
        downsamples=...,
        hidden_act=...,
        down_patch_size=...,
        down_stride=...,
        down_pad=...,
        drop_path_rate=...,
        drop_mlp_rate=...,
        drop_conv_encoder_rate=...,
        use_layer_scale=...,
        layer_scale_init_value=...,
        batch_norm_eps=...,
        **kwargs,
    ) -> None: ...

class SwiftFormerOnnxConfig(OnnxConfig):
    torch_onnx_minimum_version = ...
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]: ...
    @property
    def atol_for_validation(self) -> float: ...

__all__ = ["SwiftFormerConfig", "SwiftFormerOnnxConfig"]
