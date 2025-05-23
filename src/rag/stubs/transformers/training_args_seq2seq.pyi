"""
This type stub file was generated by pyright.
"""

from dataclasses import dataclass
from pathlib import Path

from .generation.configuration_utils import GenerationConfig
from .training_args import TrainingArguments
from .utils import add_start_docstrings

logger = ...

@dataclass
@add_start_docstrings(TrainingArguments.__doc__)
class Seq2SeqTrainingArguments(TrainingArguments):
    """
    Args:
        predict_with_generate (`bool`, *optional*, defaults to `False`):
            Whether to use generate to calculate generative metrics (ROUGE, BLEU).
        generation_max_length (`int`, *optional*):
            The `max_length` to use on each evaluation loop when `predict_with_generate=True`. Will default to the
            `max_length` value of the model configuration.
        generation_num_beams (`int`, *optional*):
            The `num_beams` to use on each evaluation loop when `predict_with_generate=True`. Will default to the
            `num_beams` value of the model configuration.
        generation_config (`str` or `Path` or [`~generation.GenerationConfig`], *optional*):
            Allows to load a [`~generation.GenerationConfig`] from the `from_pretrained` method. This can be either:

            - a string, the *model id* of a pretrained model configuration hosted inside a model repo on
              huggingface.co.
            - a path to a *directory* containing a configuration file saved using the
              [`~GenerationConfig.save_pretrained`] method, e.g., `./my_model_directory/`.
            - a [`~generation.GenerationConfig`] object.
    """

    sortish_sampler: bool = ...
    predict_with_generate: bool = ...
    generation_max_length: int | None = ...
    generation_num_beams: int | None = ...
    generation_config: str | Path | GenerationConfig | None = ...
    def to_dict(self):  # -> dict[str, Any]:
        """
        Serializes this instance while replace `Enum` by their values and `GenerationConfig` by dictionaries (for JSON
        serialization support). It obfuscates the token values by removing their value.
        """
        ...
