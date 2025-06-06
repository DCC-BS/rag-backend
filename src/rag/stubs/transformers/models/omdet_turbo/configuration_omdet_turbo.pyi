"""
This type stub file was generated by pyright.
"""

from ...configuration_utils import PretrainedConfig

"""OmDet-Turbo model configuration"""
logger = ...

class OmDetTurboConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`OmDetTurboForObjectDetection`].
    It is used to instantiate a OmDet-Turbo model according to the specified arguments, defining the model architecture
    Instantiating a configuration with the defaults will yield a similar configuration to that of the OmDet-Turbo
    [omlab/omdet-turbo-swin-tiny-hf](https://huggingface.co/omlab/omdet-turbo-swin-tiny-hf) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`PretrainedConfig`, *optional*):
            The configuration of the text backbone.
        backbone_config (`PretrainedConfig`, *optional*):
            The configuration of the vision backbone.
        use_timm_backbone (`bool`, *optional*, defaults to `True`):
            Whether to use the timm for the vision backbone.
        backbone (`str`, *optional*, defaults to `"swin_tiny_patch4_window7_224"`):
            The name of the pretrained vision backbone to use. If `use_pretrained_backbone=False` a randomly initialized
            backbone with the same architecture `backbone` is used.
        backbone_kwargs (`dict`, *optional*):
            Additional kwargs for the vision backbone.
        use_pretrained_backbone (`bool`, *optional*, defaults to `False`):
            Whether to use a pretrained vision backbone.
        apply_layernorm_after_vision_backbone (`bool`, *optional*, defaults to `True`):
            Whether to apply layer normalization on the feature maps of the vision backbone output.
        image_size (`int`, *optional*, defaults to 640):
            The size (resolution) of each image.
        disable_custom_kernels (`bool`, *optional*, defaults to `False`):
            Whether to disable custom kernels.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon value for layer normalization.
        batch_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon value for batch normalization.
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        text_projection_in_dim (`int`, *optional*, defaults to 512):
            The input dimension for the text projection.
        text_projection_out_dim (`int`, *optional*, defaults to 512):
            The output dimension for the text projection.
        task_encoder_hidden_dim (`int`, *optional*, defaults to 1024):
            The feedforward dimension for the task encoder.
        class_embed_dim (`int`, *optional*, defaults to 512):
            The dimension of the classes embeddings.
        class_distance_type (`str`, *optional*, defaults to `"cosine"`):
            The type of of distance to compare predicted classes to projected classes embeddings.
            Can be `"cosine"` or `"dot"`.
        num_queries (`int`, *optional*, defaults to 900):
            The number of queries.
        csp_activation (`str`, *optional*, defaults to `"silu"`):
            The activation function of the Cross Stage Partial (CSP) networks of the encoder.
        conv_norm_activation (`str`, *optional*, defaults to `"gelu"`):
            The activation function of the ConvNormLayer layers of the encoder.
        encoder_feedforward_activation (`str`, *optional*, defaults to `"relu"`):
            The activation function for the feedforward network of the encoder.
        encoder_feedforward_dropout (`float`, *optional*, defaults to 0.0):
            The dropout rate following the activation of the encoder feedforward network.
        encoder_dropout (`float`, *optional*, defaults to 0.0):
            The dropout rate of the encoder multi-head attention module.
        hidden_expansion (`int`, *optional*, defaults to 1):
            The hidden expansion of the CSP networks in the encoder.
        vision_features_channels (`tuple(int)`, *optional*, defaults to `[256, 256, 256]`):
            The projected vision features channels used as inputs for the decoder.
        encoder_hidden_dim (`int`, *optional*, defaults to 256):
            The hidden dimension of the encoder.
        encoder_in_channels (`List(int)`, *optional*, defaults to `[192, 384, 768]`):
            The input channels for the encoder.
        encoder_projection_indices (`List(int)`, *optional*, defaults to `[2]`):
            The indices of the input features projected by each layers.
        encoder_attention_heads (`int`, *optional*, defaults to 8):
            The number of attention heads for the encoder.
        encoder_dim_feedforward (`int`, *optional*, defaults to 2048):
            The feedforward dimension for the encoder.
        encoder_layers (`int`, *optional*, defaults to 1):
            The number of layers in the encoder.
        positional_encoding_temperature (`int`, *optional*, defaults to 10000):
            The positional encoding temperature in the encoder.
        num_feature_levels (`int`, *optional*, defaults to 3):
            The number of feature levels for the multi-scale deformable attention module of the decoder.
        decoder_hidden_dim (`int`, *optional*, defaults to 256):
            The hidden dimension of the decoder.
        decoder_num_heads (`int`, *optional*, defaults to 8):
            The number of heads for the decoder.
        decoder_num_layers (`int`, *optional*, defaults to 6):
            The number of layers for the decoder.
        decoder_activation (`str`, *optional*, defaults to `"relu"`):
            The activation function for the decoder.
        decoder_dim_feedforward (`int`, *optional*, defaults to 2048):
            The feedforward dimension for the decoder.
        decoder_num_points (`int`, *optional*, defaults to 4):
            The number of points sampled in the decoder multi-scale deformable attention module.
        decoder_dropout (`float`, *optional*, defaults to 0.0):
            The dropout rate for the decoder.
        eval_size (`Tuple[int, int]`, *optional*):
            Height and width used to computes the effective height and width of the position embeddings after taking
            into account the stride (see RTDetr).
        learn_initial_query (`bool`, *optional*, defaults to `False`):
            Whether to learn the initial query.
        cache_size (`int`, *optional*, defaults to 100):
            The cache size for the classes and prompts caches.
        is_encoder_decoder (`bool`, *optional*, defaults to `True`):
            Whether the model is used as an encoder-decoder model or not.
        kwargs (`Dict[str, Any]`, *optional*):
            Additional parameters from the architecture. The values in kwargs will be saved as part of the configuration
            and can be used to control the model outputs.

    Examples:

    ```python
    >>> from transformers import OmDetTurboConfig, OmDetTurboForObjectDetection

    >>> # Initializing a OmDet-Turbo omlab/omdet-turbo-swin-tiny-hf style configuration
    >>> configuration = OmDetTurboConfig()

    >>> # Initializing a model (with random weights) from the omlab/omdet-turbo-swin-tiny-hf style configuration
    >>> model = OmDetTurboForObjectDetection(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = ...
    attribute_map = ...
    def __init__(
        self,
        text_config=...,
        backbone_config=...,
        use_timm_backbone=...,
        backbone=...,
        backbone_kwargs=...,
        use_pretrained_backbone=...,
        apply_layernorm_after_vision_backbone=...,
        image_size=...,
        disable_custom_kernels=...,
        layer_norm_eps=...,
        batch_norm_eps=...,
        init_std=...,
        text_projection_in_dim=...,
        text_projection_out_dim=...,
        task_encoder_hidden_dim=...,
        class_embed_dim=...,
        class_distance_type=...,
        num_queries=...,
        csp_activation=...,
        conv_norm_activation=...,
        encoder_feedforward_activation=...,
        encoder_feedforward_dropout=...,
        encoder_dropout=...,
        hidden_expansion=...,
        vision_features_channels=...,
        encoder_hidden_dim=...,
        encoder_in_channels=...,
        encoder_projection_indices=...,
        encoder_attention_heads=...,
        encoder_dim_feedforward=...,
        encoder_layers=...,
        positional_encoding_temperature=...,
        num_feature_levels=...,
        decoder_hidden_dim=...,
        decoder_num_heads=...,
        decoder_num_layers=...,
        decoder_activation=...,
        decoder_dim_feedforward=...,
        decoder_num_points=...,
        decoder_dropout=...,
        eval_size=...,
        learn_initial_query=...,
        cache_size=...,
        is_encoder_decoder=...,
        **kwargs,
    ) -> None: ...

__all__ = ["OmDetTurboConfig"]
