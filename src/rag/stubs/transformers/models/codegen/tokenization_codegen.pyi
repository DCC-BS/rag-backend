"""
This type stub file was generated by pyright.
"""

from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from ...tokenization_utils import PreTrainedTokenizer

"""Tokenization classes for CodeGen"""
if TYPE_CHECKING: ...
logger = ...
VOCAB_FILES_NAMES = ...

@lru_cache
def bytes_to_unicode():  # -> dict[int, str]:
    """
    Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
    characters the bpe code barfs on.

    The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
    if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
    decent coverage. This is a significant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup
    tables between utf-8 bytes and unicode strings.
    """
    ...

def get_pairs(word):  # -> set[Any]:
    """
    Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    ...

class CodeGenTokenizer(PreTrainedTokenizer):
    """
    Construct a CodeGen tokenizer. Based on byte-level Byte-Pair-Encoding.

    This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will
    be encoded differently whether it is at the beginning of the sentence (without space) or not:

    ```python
    >>> from transformers import CodeGenTokenizer

    >>> tokenizer = CodeGenTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
    >>> tokenizer("Hello world")["input_ids"]
    [15496, 995]

    >>> tokenizer(" Hello world")["input_ids"]
    [18435, 995]
    ```

    You can get around that behavior by passing `add_prefix_space=True` when instantiating this tokenizer or when you
    call it on some text, but since the model was not pretrained this way, it might yield a decrease in performance.

    <Tip>

    When used with `is_split_into_words=True`, this tokenizer will add a space before each word (even the first one).

    </Tip>

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        merges_file (`str`):
            Path to the merges file.
        errors (`str`, *optional*, defaults to `"replace"`):
            Paradigm to follow when decoding bytes to UTF-8. See
            [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.
        unk_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The beginning of sequence token.
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The end of sequence token.
        pad_token (`str`, *optional*):
            The token used for padding, for example when batching sequences of different lengths.
        add_prefix_space (`bool`, *optional*, defaults to `False`):
            Whether or not to add an initial space to the input. This allows to treat the leading word just as any
            other word. (CodeGen tokenizer detect beginning of words by the preceding space).
        add_bos_token (`bool`, *optional*, defaults to `False`):
            Whether to add a beginning of sequence token at the start of sequences.
        return_token_type_ids (`bool`, *optional*, defaults to `False`):
            Whether to return token type IDs.
    """

    vocab_files_names = ...
    model_input_names = ...
    def __init__(
        self,
        vocab_file,
        merges_file,
        errors=...,
        unk_token=...,
        bos_token=...,
        eos_token=...,
        pad_token=...,
        add_prefix_space=...,
        add_bos_token=...,
        return_token_type_ids=...,
        **kwargs,
    ) -> None: ...
    @property
    def vocab_size(self):  # -> int:
        ...
    def get_vocab(self):  # -> dict[str, Any]:
        ...
    def bpe(self, token):  # -> LiteralString:
        ...
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=...):  # -> list[str | list[str] | Any | None]:
        ...
    def convert_tokens_to_string(self, tokens):  # -> str:
        """Converts a sequence of tokens (string) in a single string."""
        ...

    def create_token_type_ids_from_sequences(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = ...
    ) -> list[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A sequence
        pair mask has the following format:

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```

        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        ...

    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = ...) -> tuple[str]: ...
    def prepare_for_tokenization(self, text, is_split_into_words=..., **kwargs):  # -> tuple[str, dict[str, Any]]:
        ...
    def decode(
        self,
        token_ids: int | list[int] | np.ndarray | torch.Tensor | tf.Tensor,
        skip_special_tokens: bool = ...,
        clean_up_tokenization_spaces: bool = ...,
        truncate_before_pattern: list[str] | None = ...,
        **kwargs,
    ) -> str:
        """
        Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special
        tokens and clean up tokenization spaces.

        Similar to doing `self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))`.

        Args:
            token_ids (`Union[int, List[int], np.ndarray, torch.Tensor, tf.Tensor]`):
                List of tokenized input ids. Can be obtained using the `__call__` method.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.
            clean_up_tokenization_spaces (`bool`, *optional*):
                Whether or not to clean up the tokenization spaces. If `None`, will default to
                `self.clean_up_tokenization_spaces` (available in the `tokenizer_config`).
            truncate_before_pattern (`List[str]`, *optional*, defaults to `None`):
                A list of regular expression strings that will be used to truncate the returned string. This can be
                used to remove extra pieces of code (e.g. truncate if observing a comment symbol "#" at the beginning
                of a new line). An example pattern could be `["^#", re.escape("<|endoftext|>"), "^'''", "\n\n\n"]`.
            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific decode method.

        Returns:
            `str`: The decoded sentence.
        """
        ...

    def truncate(self, completion, truncate_before_pattern): ...

__all__ = ["CodeGenTokenizer"]
