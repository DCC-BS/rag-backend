"""
This type stub file was generated by pyright.
"""

from ...tokenization_utils_fast import PreTrainedTokenizerFast

"""Fast Tokenization class for model DeBERTa."""
logger = ...
VOCAB_FILES_NAMES = ...

class DebertaTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" DeBERTa tokenizer (backed by HuggingFace's *tokenizers* library). Based on byte-level
    Byte-Pair-Encoding.

    This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will
    be encoded differently whether it is at the beginning of the sentence (without space) or not:

    ```python
    >>> from transformers import DebertaTokenizerFast

    >>> tokenizer = DebertaTokenizerFast.from_pretrained("microsoft/deberta-base")
    >>> tokenizer("Hello world")["input_ids"]
    [1, 31414, 232, 2]

    >>> tokenizer(" Hello world")["input_ids"]
    [1, 20920, 232, 2]
    ```

    You can get around that behavior by passing `add_prefix_space=True` when instantiating this tokenizer, but since
    the model was not pretrained this way, it might yield a decrease in performance.

    <Tip>

    When used with `is_split_into_words=True`, this tokenizer needs to be instantiated with `add_prefix_space=True`.

    </Tip>

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`, *optional*):
            Path to the vocabulary file.
        merges_file (`str`, *optional*):
            Path to the merges file.
        tokenizer_file (`str`, *optional*):
            The path to a tokenizer file to use instead of the vocab file.
        errors (`str`, *optional*, defaults to `"replace"`):
            Paradigm to follow when decoding bytes to UTF-8. See
            [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.
        bos_token (`str`, *optional*, defaults to `"[CLS]"`):
            The beginning of sequence token.
        eos_token (`str`, *optional*, defaults to `"[SEP]"`):
            The end of sequence token.
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        unk_token (`str`, *optional*, defaults to `"[UNK]"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"[PAD]"`):
            The token used for padding, for example when batching sequences of different lengths.
        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        add_prefix_space (`bool`, *optional*, defaults to `False`):
            Whether or not to add an initial space to the input. This allows to treat the leading word just as any
            other word. (Deberta tokenizer detect beginning of words by the preceding space).
    """

    vocab_files_names = ...
    model_input_names = ...
    slow_tokenizer_class = ...
    def __init__(
        self,
        vocab_file=...,
        merges_file=...,
        tokenizer_file=...,
        errors=...,
        bos_token=...,
        eos_token=...,
        sep_token=...,
        cls_token=...,
        unk_token=...,
        pad_token=...,
        mask_token=...,
        add_prefix_space=...,
        **kwargs,
    ) -> None: ...
    @property
    def mask_token(self) -> str:
        """
        `str`: Mask token, to use when training a model with masked-language modeling. Log an error if used while not
        having been set.

        Deberta tokenizer has a special mask token to be used in the fill-mask pipeline. The mask token will greedily
        comprise the space before the *[MASK]*.
        """
        ...

    @mask_token.setter
    def mask_token(self, value):  # -> None:
        """
        Overriding the default behavior of the mask token to have it eat the space before it.
        """
        ...

    def build_inputs_with_special_tokens(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = ...
    ) -> list[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A DeBERTa sequence has the following format:

        - single sequence: [CLS] X [SEP]
        - pair of sequences: [CLS] A [SEP] B [SEP]

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        ...

    def create_token_type_ids_from_sequences(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = ...
    ) -> list[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A DeBERTa
        sequence pair mask has the following format:

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

__all__ = ["DebertaTokenizerFast"]
