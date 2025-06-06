"""
This type stub file was generated by pyright.
"""

from collections.abc import Callable

from ...image_processing_utils import BaseImageProcessor
from ...image_utils import ImageInput
from ...utils import TensorType

"""Image processor class for Idefics."""
IDEFICS_STANDARD_MEAN = ...
IDEFICS_STANDARD_STD = ...

def convert_to_rgb(image):  # -> Image:
    ...

class IdeficsImageProcessor(BaseImageProcessor):
    r"""
    Constructs a Idefics image processor.

    Args:
        image_size (`int`, *optional*, defaults to 224):
            Resize to image size
        image_mean (`float` or `List[float]`, *optional*, defaults to `IDEFICS_STANDARD_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method. Can be
            overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IDEFICS_STANDARD_STD`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
            Can be overridden by the `image_std` parameter in the `preprocess` method.
        image_num_channels (`int`, *optional*, defaults to 3):
            Number of image channels.
    """

    model_input_names = ...
    def __init__(
        self,
        image_size: int = ...,
        image_mean: float | list[float] | None = ...,
        image_std: float | list[float] | None = ...,
        image_num_channels: int | None = ...,
        **kwargs,
    ) -> None: ...
    def preprocess(
        self,
        images: ImageInput,
        image_num_channels: int | None = ...,
        image_size: dict[str, int] | None = ...,
        image_mean: float | list[float] | None = ...,
        image_std: float | list[float] | None = ...,
        transform: Callable = ...,
        return_tensors: str | TensorType | None = ...,
        **kwargs,
    ) -> TensorType:
        """
        Preprocess a batch of images.

        Args:
            images (`ImageInput`):
                A list of images to preprocess.
            image_size (`int`, *optional*, defaults to `self.image_size`):
                Resize to image size
            image_num_channels (`int`, *optional*, defaults to `self.image_num_channels`):
                Number of image channels.
            image_mean (`float` or `List[float]`, *optional*, defaults to `IDEFICS_STANDARD_MEAN`):
                Mean to use if normalizing the image. This is a float or list of floats the length of the number of
                channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method. Can
                be overridden by the `image_mean` parameter in the `preprocess` method.
            image_std (`float` or `List[float]`, *optional*, defaults to `IDEFICS_STANDARD_STD`):
                Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
                number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess`
                method. Can be overridden by the `image_std` parameter in the `preprocess` method.
            transform (`Callable`, *optional*, defaults to `None`):
                A custom transform function that accepts a single image can be passed for training. For example,
                `torchvision.Compose` can be used to compose multiple transforms. If `None` - an inference mode is
                assumed - and then a preset of inference-specific transforms will be applied to the images

        Returns:
            a PyTorch tensor of the processed images

        """
        ...

__all__ = ["IdeficsImageProcessor"]
