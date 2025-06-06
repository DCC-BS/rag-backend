"""
This type stub file was generated by pyright.
"""

from typing import NamedTuple

import numpy as np

from .utils import is_torch_available

if is_torch_available(): ...
logger = ...
GGUF_TO_TRANSFORMERS_MAPPING = ...
GGUF_SUPPORTED_ARCHITECTURES = ...

class GGUFTensor(NamedTuple):
    weights: np.ndarray
    name: str
    metadata: dict
    ...

class TensorProcessor:
    def __init__(self, config=...) -> None: ...
    def process(self, weights, name, **kwargs):  # -> GGUFTensor:
        ...

class LlamaTensorProcessor(TensorProcessor):
    def __init__(self, config=...) -> None: ...
    def process(self, weights, name, **kwargs):  # -> GGUFTensor:
        ...

class Qwen2MoeTensorProcessor(TensorProcessor):
    def __init__(self, config=...) -> None: ...
    def process(self, weights, name, **kwargs):  # -> GGUFTensor:
        ...

class BloomTensorProcessor(TensorProcessor):
    def __init__(self, config=...) -> None: ...
    def process(self, weights, name, **kwargs):  # -> GGUFTensor:
        ...

class T5TensorProcessor(TensorProcessor):
    def __init__(self, config=...) -> None: ...
    def process(self, weights, name, **kwargs):  # -> GGUFTensor:
        ...

class GPT2TensorProcessor(TensorProcessor):
    def __init__(self, config=...) -> None: ...
    def process(self, weights, name, **kwargs):  # -> GGUFTensor:
        ...

class MambaTensorProcessor(TensorProcessor):
    def __init__(self, config=...) -> None: ...
    def process(self, weights, name, **kwargs):  # -> GGUFTensor:
        ...

class NemotronTensorProcessor(TensorProcessor):
    def __init__(self, config=...) -> None: ...
    def process(self, weights, name, **kwargs):  # -> GGUFTensor:
        ...

class Gemma2TensorProcessor(TensorProcessor):
    def __init__(self, config=...) -> None: ...
    def process(self, weights, name, **kwargs):  # -> GGUFTensor:
        ...

TENSOR_PROCESSORS = ...

def read_field(reader, field):  # -> list[int | float | bool | str | Any]:
    ...
def get_gguf_hf_weights_map(
    hf_model, model_type: str | None = ..., num_layers: int | None = ..., qual_name: str = ...
):  # -> dict[Any, Any]:
    """
    GGUF uses this naming convention for their tensors from HF checkpoint:
    `blk.N.BB.weight` and `blk.N.BB.bias`
    where N signifies the block number of a layer, and BB signifies the
    attention/mlp layer components.
    See "Standardized tensor names" in
    https://github.com/ggerganov/ggml/blob/master/docs/gguf.md for details.
    """
    ...

def load_gguf_checkpoint(gguf_checkpoint_path, return_tensors=..., model_to_load=...):  # -> dict[str, dict[Any, Any]]:
    """
    Load a GGUF file and return a dictionary of parsed parameters containing tensors, the parsed
    tokenizer and config attributes.

    Args:
        gguf_checkpoint_path (`str`):
            The path the to GGUF file to load
        return_tensors (`bool`, defaults to `True`):
            Whether to read the tensors from the file and return them. Not doing so is faster
            and only loads the metadata in memory.
    """
    ...
