"""
This type stub file was generated by pyright.
"""

from functools import lru_cache

from ...tokenization_utils import PreTrainedTokenizer

"""Tokenization classes for OpenAI GPT."""
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

class GPT2Tokenizer(PreTrainedTokenizer):
    """
    Construct a GPT-2 tokenizer. Based on byte-level Byte-Pair-Encoding.

    This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will
    be encoded differently whether it is at the beginning of the sentence (without space) or not:

    ```python
    >>> from transformers import GPT2Tokenizer

    >>> tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
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
            other word. (GPT2 tokenizer detect beginning of words by the preceding space).
        add_bos_token (`bool`, *optional*, defaults to `False`):
            Whether or not to add an initial beginning of sentence token to the input. This allows to treat the leading
            word just as any other word.
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
    def get_special_tokens_mask(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = ..., already_has_special_tokens: bool = ...
    ) -> list[int]:
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` or `encode_plus` methods.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        ...

    def convert_tokens_to_string(self, tokens):  # -> str:
        """Converts a sequence of tokens (string) in a single string."""
        ...

    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = ...) -> tuple[str]: ...
    def prepare_for_tokenization(self, text, is_split_into_words=..., **kwargs):  # -> tuple[str, dict[str, Any]]:
        ...

__all__ = ["GPT2Tokenizer"]
