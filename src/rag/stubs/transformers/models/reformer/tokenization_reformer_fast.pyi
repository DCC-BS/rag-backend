"""
This type stub file was generated by pyright.
"""

from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import is_sentencepiece_available
from .tokenization_reformer import ReformerTokenizer

"""Tokenization class for model Reformer."""
if is_sentencepiece_available(): ...
else:
    ReformerTokenizer = ...
logger = ...
SPIECE_UNDERLINE = ...
VOCAB_FILES_NAMES = ...

class ReformerTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" Reformer tokenizer (backed by HuggingFace's *tokenizers* library). Based on
    [Unigram](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=unigram#models).

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a *.spm* extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the end of sequence.
            The token used is the `sep_token`.

            </Tip>

        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        additional_special_tokens (`List[str]`, *optional*):
            Additional special tokens used by the tokenizer.
    """

    vocab_files_names = ...
    model_input_names = ...
    slow_tokenizer_class = ...
    def __init__(
        self, vocab_file=..., tokenizer_file=..., eos_token=..., unk_token=..., additional_special_tokens=..., **kwargs
    ) -> None: ...
    @property
    def can_save_slow_tokenizer(self) -> bool: ...
    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = ...) -> tuple[str]: ...

__all__ = ["ReformerTokenizerFast"]
