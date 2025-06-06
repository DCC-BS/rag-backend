"""
This type stub file was generated by pyright.
"""

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, TextKwargs, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput

"""
Processor class for Chameleon.
"""

class ChameleonTextKwargs(TextKwargs, total=False):
    return_for_text_completion: bool
    ...

class ChameleonProcessorKwargs(ProcessingKwargs, total=False):
    text_kwargs: ChameleonTextKwargs
    _defaults = ...

class ChameleonProcessor(ProcessorMixin):
    r"""
    Constructs a Chameleon processor which wraps a Chameleon image processor and a Chameleon tokenizer into a single
    processor.

    [`ChameleonProcessor`] offers all the functionalities of [`ChameleonImageProcessor`] and [`LlamaTokenizerFast`].
    See the [`~ChameleonProcessor.__call__`] and [`~ChameleonProcessor.decode`] for more information.

    Args:
        image_processor ([`ChameleonImageProcessor`]):
            The image processor is a required input.
        tokenizer ([`LlamaTokenizerFast`]):
            The tokenizer is a required input.
        image_seq_length (`int`, *optional*, defaults to 1024):
            Sequence length of one image embedding.
        image_token (`str`, *optional*, defaults to `"<image>"`):
            The special token used to indicate image in the text.
    """

    attributes = ...
    tokenizer_class = ...
    valid_kwargs = ...
    image_processor_class = ...
    def __init__(self, image_processor, tokenizer, image_seq_length: int = ..., image_token: str = ...) -> None: ...
    def __call__(
        self,
        images: ImageInput | None = ...,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = ...,
        audio=...,
        videos=...,
        **kwargs: Unpack[ChameleonProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to LlamaTokenizerFast's [`~LlamaTokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` and `kwrags` arguments to
        CLIPImageProcessor's [`~CLIPImageProcessor.__call__`] if `images` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
        """
        ...

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        ...

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        ...

    @property
    def model_input_names(self):  # -> list[Any]:
        ...

__all__ = ["ChameleonProcessor"]
