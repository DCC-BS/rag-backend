"""
This type stub file was generated by pyright.
"""

from dataclasses import dataclass

import torch
from mambapy.pscan import pscan
from torch import nn

from ...cache_utils import MambaCache
from ...generation import GenerationMixin
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)
from ...utils.import_utils import is_causal_conv1d_available, is_mamba_ssm_available, is_mambapy_available
from .configuration_mamba import MambaConfig

"""PyTorch MAMBA model."""
logger = ...
if is_mambapy_available(): ...
else:
    pscan = ...
if is_mamba_ssm_available(): ...
else: ...
if is_causal_conv1d_available(): ...
else: ...
is_fast_path_available = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...

class MambaMixer(nn.Module):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute the `contextualized_states`.
    A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
    ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
    and is why Mamba is called **selective** state spaces)
    """
    def __init__(self, config: MambaConfig, layer_idx: int) -> None: ...
    def cuda_kernels_forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: MambaCache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        attention_mask: torch.LongTensor | None = ...,
    ):  # -> Any:
        ...
    def slow_forward(
        self,
        input_states,
        cache_params: MambaCache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        attention_mask: torch.LongTensor | None = ...,
    ):  # -> Any:
        ...
    def forward(
        self,
        hidden_states,
        cache_params: MambaCache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        attention_mask: torch.LongTensor | None = ...,
    ):  # -> Any:
        ...

class MambaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=...) -> None:
        """
        MambaRMSNorm is equivalent to T5LayerNorm and LlamaRMSNorm
        """
        ...

    def forward(self, hidden_states): ...
    def extra_repr(self):  # -> str:
        ...

class MambaBlock(nn.Module):
    def __init__(self, config, layer_idx) -> None: ...
    def forward(
        self,
        hidden_states,
        cache_params: MambaCache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        attention_mask: torch.LongTensor | None = ...,
    ): ...

class MambaPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = MambaConfig
    base_model_prefix = ...
    _no_split_modules = ...
    supports_gradient_checkpointing = ...
    _is_stateful = ...

@dataclass
class MambaOutput(ModelOutput):
    """
    Class for the MAMBA model outputs.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        cache_params (`MambaCache`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.

            Includes both the State space model state matrices after the selective scan, and the Convolutional states
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    last_hidden_state: torch.FloatTensor | None = ...
    cache_params: MambaCache | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...

@dataclass
class MambaCausalLMOutput(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        cache_params (`MambaCache`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.

            Includes both the State space model state matrices after the selective scan, and the Convolutional states
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    loss: torch.FloatTensor | None = ...
    logits: torch.FloatTensor | None = ...
    cache_params: MambaCache | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...

MAMBA_START_DOCSTRING = ...
MAMBA_INPUTS_DOCSTRING = ...

@add_start_docstrings(
    "The bare MAMBA Model transformer outputting raw hidden-states without any specific head on top.",
    MAMBA_START_DOCSTRING,
)
class MambaModel(MambaPreTrainedModel):
    def __init__(self, config) -> None: ...
    def load_hook(self, state_dict, prefix, *args):  # -> None:
        ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, new_embeddings):  # -> None:
        ...
    @add_start_docstrings_to_model_forward(MAMBA_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=MambaOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        inputs_embeds: torch.LongTensor | None = ...,
        cache_params: MambaCache | None = ...,
        use_cache: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        attention_mask: torch.LongTensor | None = ...,
    ) -> tuple | MambaOutput: ...

@add_start_docstrings(
    """
    The MAMBA Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    MAMBA_START_DOCSTRING,
)
class MambaForCausalLM(MambaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ...
    def __init__(self, config) -> None: ...
    def get_output_embeddings(self):  # -> Linear:
        ...
    def set_output_embeddings(self, new_embeddings):  # -> None:
        ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, new_embeddings):  # -> None:
        ...
    def prepare_inputs_for_generation(
        self,
        input_ids,
        inputs_embeds=...,
        use_cache=...,
        cache_params: MambaCache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        attention_mask: torch.LongTensor | None = ...,
        **kwargs,
    ):  # -> dict[str, Any]:
        ...
    @add_start_docstrings_to_model_forward(MAMBA_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC, output_type=MambaCausalLMOutput, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.LongTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        cache_params: MambaCache | None = ...,
        labels: torch.LongTensor | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        use_cache: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
        **kwargs,
    ) -> tuple | MambaCausalLMOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        ...

__all__ = ["MambaForCausalLM", "MambaModel", "MambaPreTrainedModel"]
