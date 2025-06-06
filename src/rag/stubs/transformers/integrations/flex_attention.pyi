"""
This type stub file was generated by pyright.
"""

import torch
from torch.nn.attention.flex_attention import BlockMask

from ..utils import is_torch_flex_attn_available

"""
Partially inspired by torchtune's flex attention implementation

Citation:
@software{torchtune,
  title = {torchtune: PyTorch's finetuning library},
  author = {torchtune maintainers and contributors},
  url = {https//github.com/pytorch/torchtune},
  license = {BSD-3-Clause},
  month = apr,
  year = {2024}
}
"""
if is_torch_flex_attn_available(): ...

class WrappedFlexAttention:
    """
    We are doing a singleton class so that flex attention is compiled once when it's first called.
    """

    _instance = ...
    _is_flex_compiled = ...
    _compiled_flex_attention = ...
    def __new__(cls, *args, **kwargs):  # -> Self:
        ...
    @torch.compiler.disable(recursive=False)
    def __init__(self) -> None:
        """
        Initialize or update the singleton instance.
        """
        ...

    def __call__(self):  # -> Callable[..., Tensor | Tuple[Tensor, Tensor]] | None:
        ...

def make_flex_block_causal_mask(attention_mask_2d: torch.Tensor) -> BlockMask:
    """
    Create a block causal document mask for a batch of sequences, both packed and unpacked.
    Create Block causal logic and passing it into :func:`torch.nn.attention.flex_attention.create_block_mask`.
    The resultant BlockMask is a compressed representation of the full block causal
    mask. BlockMask is essential for performant computation of flex attention.
    See: https://pytorch.org/blog/flexattention/

    Args:
        attention_mask_2d (torch.Tensor): Attention mask for packed and padded sequences
        of shape (batch_size, total_seq_len). e.g.

        For unpacked sequence:
        [[1, 1, 1, 1, 0, 0, 0],
         [1, 1, 1, 1, 1, 0, 0]]

        For packed sequence:
        [[1, 1, 1, 2, 2, 2, 0],
         [1, 1, 2, 2, 2, 3, 3]]

    Returns:
        BlockMask
    """
    ...

@torch.compiler.disable(recursive=False)
def compile_friendly_flex_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, **kwargs
) -> torch.Tensor: ...
def flex_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | BlockMask,
    scaling: float | None = ...,
    softcap: float | None = ...,
    head_mask: torch.Tensor | None = ...,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]: ...
