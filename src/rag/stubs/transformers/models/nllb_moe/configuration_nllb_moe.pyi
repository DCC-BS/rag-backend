"""
This type stub file was generated by pyright.
"""

from ...configuration_utils import PretrainedConfig

"""NLLB-MoE model configuration"""
logger = ...

class NllbMoeConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`NllbMoeModel`]. It is used to instantiate an
    NLLB-MoE model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the NLLB-MoE
    [facebook/nllb-moe-54b](https://huggingface.co/facebook/nllb-moe-54b) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 50265):
            Vocabulary size of the NllbMoe model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`NllbMoeModel`] or
        d_model (`int`, *optional*, defaults to 1024):
            Dimensionality of the layers and the pooler layer.
        encoder_layers (`int`, *optional*, defaults to 12):
            Number of encoder layers.
        decoder_layers (`int`, *optional*, defaults to 12):
            Number of decoder layers.
        encoder_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        decoder_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer decoder.
        decoder_ffn_dim (`int`, *optional*, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
        encoder_ffn_dim (`int`, *optional*, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in encoder.
        activation_function (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        classifier_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for classifier.
        max_position_embeddings (`int`, *optional*, defaults to 1024):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        encoder_layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the encoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
            for more details.
        decoder_layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the decoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
            for more details.
        second_expert_policy ( `str`, *optional*, default to `"all"`):
            The policy used for the sampling the probability of being sampled to a second expert for each token.
        normalize_router_prob_before_dropping (`bool`, *optional*, defaults to `True`):
            Whether or not to normalize the router probabilities before applying a mask based on the experts capacity
            (capacity dropping).
        batch_prioritized_routing (`bool`, *optional*, defaults to `True`):
            Whether or not to orders the tokens by their router probabilities before capacity dropping. This means that
            the tokens that have the highest probabilities will be routed before other tokens that might be further in
            the sequence.
        moe_eval_capacity_token_fraction (`float`, *optional*, defaults to 1.0):
            Fraction of tokens as capacity during validation, if set to negative, uses the same as training. Should be
            in range: (0.0, 1.0].
        num_experts (`int`, *optional*, defaults to 128):
            Number of experts for each NllbMoeSparseMlp layer.
        expert_capacity (`int`, *optional*, defaults to 64):
            Number of tokens that can be stored in each expert.
        encoder_sparse_step (`int`, *optional*, defaults to 4):
            Frequency of the sparse layers in the encoder. 4 means that one out of 4 layers will be sparse.
        decoder_sparse_step (`int`, *optional*, defaults to 4):
            Frequency of the sparse layers in the decoder. 4 means that one out of 4 layers will be sparse.
        router_dtype (`str`, *optional*, default to `"float32"`):
            The `dtype` used for the routers. It is preferable to keep the `dtype` to `"float32"` as specified in the
            *selective precision* discussion in [the paper](https://arxiv.org/abs/2101.03961).
        router_ignore_padding_tokens (`bool`, *optional*, defaults to `False`):
            Whether to ignore padding tokens when routing. if `False`, the padding tokens are not routed to any
            experts.
        router_bias (`bool`, *optional*, defaults to `False`):
            Whether or not the classifier of the router should have a bias.
        moe_token_dropout (`float`, *optional*, default to 0.2):
            Masking rate for MoE expert output masking (EOM), which is implemented via a Dropout2d on the expert
            outputs.
        output_router_logits (`bool`, *optional*, defaults to `False`):
            Whether or not to return the router logits. Only set to `True` to get the auxiliary loss when training.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).

    Example:

    ```python
    >>> from transformers import NllbMoeModel, NllbMoeConfig

    >>> # Initializing a NllbMoe facebook/nllb-moe-54b style configuration
    >>> configuration = NllbMoeConfig()

    >>> # Initializing a model from the facebook/nllb-moe-54b style configuration
    >>> model = NllbMoeModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = ...
    keys_to_ignore_at_inference = ...
    attribute_map = ...
    def __init__(
        self,
        vocab_size=...,
        max_position_embeddings=...,
        encoder_layers=...,
        encoder_ffn_dim=...,
        encoder_attention_heads=...,
        decoder_layers=...,
        decoder_ffn_dim=...,
        decoder_attention_heads=...,
        encoder_layerdrop=...,
        decoder_layerdrop=...,
        use_cache=...,
        is_encoder_decoder=...,
        activation_function=...,
        d_model=...,
        dropout=...,
        attention_dropout=...,
        activation_dropout=...,
        init_std=...,
        decoder_start_token_id=...,
        scale_embedding=...,
        router_bias=...,
        router_dtype=...,
        router_ignore_padding_tokens=...,
        num_experts=...,
        expert_capacity=...,
        encoder_sparse_step=...,
        decoder_sparse_step=...,
        router_z_loss_coef=...,
        router_aux_loss_coef=...,
        second_expert_policy=...,
        normalize_router_prob_before_dropping=...,
        batch_prioritized_routing=...,
        moe_eval_capacity_token_fraction=...,
        moe_token_dropout=...,
        pad_token_id=...,
        bos_token_id=...,
        eos_token_id=...,
        output_router_logits=...,
        **kwargs,
    ) -> None: ...

__all__ = ["NllbMoeConfig"]
