"""
This type stub file was generated by pyright.
"""

import torch
from torch import Tensor

from ...modeling_outputs import BackboneOutput
from ...modeling_utils import PreTrainedModel
from ...utils import is_timm_available, is_torch_available
from ...utils.backbone_utils import BackboneMixin
from .configuration_timm_backbone import TimmBackboneConfig

if is_timm_available(): ...
if is_torch_available(): ...

class TimmBackbone(PreTrainedModel, BackboneMixin):
    """
    Wrapper class for timm models to be used as backbones. This enables using the timm models interchangeably with the
    other models in the library keeping the same API.
    """

    main_input_name = ...
    supports_gradient_checkpointing = ...
    config_class = TimmBackboneConfig
    def __init__(self, config, **kwargs) -> None: ...
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):  # -> Self:
        ...
    def freeze_batch_norm_2d(self):  # -> None:
        ...
    def unfreeze_batch_norm_2d(self):  # -> None:
        ...
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        **kwargs,
    ) -> BackboneOutput | tuple[Tensor, ...]: ...

__all__ = ["TimmBackbone"]
