"""
This type stub file was generated by pyright.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import smdistributed.modelparallel.torch as smp
import torch
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import Dataset, IterableDataset, Sampler
from torch.utils.data.distributed import DistributedSampler

from .utils import is_sagemaker_mp_enabled, is_torch_available, is_torch_xla_available, is_training_run_on_sagemaker

"""
Torch utilities for the Trainer class.
"""
if is_training_run_on_sagemaker(): ...
if is_torch_xla_available(): ...
if is_torch_available(): ...
logger = ...

def get_dataloader_sampler(dataloader):  # -> None:
    ...
def atleast_1d(tensor_or_array: torch.Tensor | np.ndarray):  # -> Tensor | ndarray[Any, Any]:
    ...
def torch_pad_and_concatenate(tensor1, tensor2, padding_index=...):  # -> Tensor:
    """Concatenates `tensor1` and `tensor2` on first axis, applying padding on the second if necessary."""
    ...

def numpy_pad_and_concatenate(array1, array2, padding_index=...):  # -> NDArray[Any]:
    """Concatenates `array1` and `array2` on first axis, applying padding on the second if necessary."""
    ...

def nested_concat(
    tensors, new_tensors, padding_index=...
):  # -> list[Any] | tuple[Any, ...] | Tensor | Mapping[Any, object] | NDArray[Any]:
    """
    Concat the `new_tensors` to `tensors` on the first dim and pad them on the second if needed. Works for tensors or
    nested list/tuples/dict of tensors.
    """
    ...

def find_batch_size(tensors):  # -> int | None:
    """
    Find the first dimension of a tensor in a nested list/tuple/dict of tensors.
    """
    ...

def nested_numpify(tensors):  # -> list[Any] | tuple[Any, ...] | Mapping[Any, object]:
    "Numpify `tensors` (even if it's a nested list/tuple/dict of tensors)."
    ...

def nested_detach(tensors):  # -> list[Any] | tuple[Any, ...] | Mapping[Any, object] | Tensor:
    "Detach `tensors` (even if it's a nested list/tuple/dict of tensors)."
    ...

def nested_xla_mesh_reduce(tensors, name):  # -> list[Any] | tuple[Any, ...] | Mapping[Any, object]:
    ...
def distributed_concat(tensor: Any, num_total_examples: int | None = ...) -> Any: ...
def distributed_broadcast_scalars(
    scalars: list[int | float], num_total_examples: int | None = ..., device: torch.device | None = ...
) -> torch.Tensor: ...
def reissue_pt_warnings(caught_warnings):  # -> None:
    ...
@contextmanager
def torch_distributed_zero_first(local_rank: int):  # -> Generator[None, Any, None]:
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.

    Args:
        local_rank (`int`): The rank of the local process.
    """
    ...

class DistributedSamplerWithLoop(DistributedSampler):
    """
    Like a torch.utils.data.distributed.DistributedSampler` but loops at the end back to the beginning of the shuffled
    samples to make each process have a round multiple of batch_size samples.

    Args:
        dataset (`torch.utils.data.Dataset`):
            Dataset used for sampling.
        batch_size (`int`):
            The batch size used with this sampler
        kwargs (`Dict[str, Any]`, *optional*):
            All other keyword arguments passed to `DistributedSampler`.
    """
    def __init__(self, dataset, batch_size, **kwargs) -> None: ...
    def __iter__(self):  # -> Iterator[Any]:
        ...

class EvalLoopContainer:
    """
    Container to store intermediate results of evaluation loop

    Args:
        do_nested_concat (`bool`, *optional*, defaults to `True`):
            If set to `True`, each iteration will recursively concatenate a new object containing tensors to
            the existing stored tensors, provided that the structure of the existing object and the new one
            are identical. If set to `False`, all newly added tensors will be stored in a list.
        padding_index (`int`, *optional*, defaults to -100):
            Value used to pad tensors of different shapes when `do_nested_concat=True`.
    """
    def __init__(self, do_nested_concat: bool = ..., padding_index: int = ...) -> None: ...
    def add(self, tensors) -> None:
        """Add tensors to the stored objects. If `do_nested_concat=True`, the tensors will be concatenated recursively."""
        ...

    def to_cpu_and_numpy(self) -> None:
        """Move tensors in stored objects to CPU and convert them to numpy arrays."""
        ...

    def get_arrays(
        self,
    ):  # -> list[Any] | tuple[Any, ...] | Mapping[Any, object] | Mapping[Any, Any] | ndarray[Any, Any] | Tensor | NDArray[Any] | None:
        """Returns the numpified and moved to CPU stored objects."""
        ...

class SequentialDistributedSampler(Sampler):
    """
    Distributed Sampler that subsamples indices sequentially, making it easier to collate all results at the end.

    Even though we only use this sampler for eval and predict (no training), which means that the model params won't
    have to be synced (i.e. will not hang for synchronization even if varied number of forward passes), we still add
    extra samples to the sampler to make it evenly divisible (like in `DistributedSampler`) to make it easy to `gather`
    or `reduce` resulting tensors at the end of the loop.
    """
    def __init__(self, dataset, num_replicas=..., rank=..., batch_size=...) -> None: ...
    def __iter__(self):  # -> Iterator[int]:
        ...
    def __len__(self):  # -> int:
        ...

def get_tpu_sampler(dataset: torch.utils.data.Dataset, batch_size: int):  # -> RandomSampler | DistributedSampler[Any]:
    ...
def nested_new_like(arrays, num_samples, padding_index=...):  # -> list[Any] | tuple[Any, ...] | NDArray[Any]:
    """Create the same nested structure as `arrays` with a first dimension always at `num_samples`."""
    ...

def expand_like(arrays, new_seq_length, padding_index=...):
    """Expand the `arrays` so that the second dimension grows to `new_seq_length`. Uses `padding_index` for padding."""
    ...

def nested_truncate(tensors, limit):  # -> list[Any] | tuple[Any, ...] | Mapping[Any, object]:
    "Truncate `tensors` at `limit` (even if it's a nested list/tuple/dict of tensors)."
    ...

class DistributedTensorGatherer:
    """
    A class responsible for properly gathering tensors (or nested list/tuple of tensors) on the CPU by chunks.

    If our dataset has 16 samples with a batch size of 2 on 3 processes and we gather then transfer on CPU at every
    step, our sampler will generate the following indices:

        `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1]`

    to get something of size a multiple of 3 (so that each process gets the same dataset length). Then process 0, 1 and
    2 will be responsible of making predictions for the following samples:

        - P0: `[0, 1, 2, 3, 4, 5]`
        - P1: `[6, 7, 8, 9, 10, 11]`
        - P2: `[12, 13, 14, 15, 0, 1]`

    The first batch treated on each process will be

        - P0: `[0, 1]`
        - P1: `[6, 7]`
        - P2: `[12, 13]`

    So if we gather at the end of the first batch, we will get a tensor (nested list/tuple of tensor) corresponding to
    the following indices:

        `[0, 1, 6, 7, 12, 13]`

    If we directly concatenate our results without taking any precautions, the user will then get the predictions for
    the indices in this order at the end of the prediction loop:

        `[0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15, 4, 5, 10, 11, 0, 1]`

    For some reason, that's not going to roll their boat. This class is there to solve that problem.

    Args:
        world_size (`int`):
            The number of processes used in the distributed training.
        num_samples (`int`):
            The number of samples in our dataset.
        make_multiple_of (`int`, *optional*):
            If passed, the class assumes the datasets passed to each process are made to be a multiple of this argument
            (by adding samples).
        padding_index (`int`, *optional*, defaults to -100):
            The padding index to use if the arrays don't all have the same sequence length.
    """
    def __init__(self, world_size, num_samples, make_multiple_of=..., padding_index=...) -> None: ...
    def add_arrays(self, arrays):  # -> None:
        """
        Add `arrays` to the internal storage, Will initialize the storage to the full size at the first arrays passed
        so that if we're bound to get an OOM, it happens at the beginning.
        """
        ...

    def finalize(self):  # -> list[Any] | tuple[Any, ...] | Mapping[Any, Any] | ndarray[_Shape, dtype[Any]] | None:
        """
        Return the properly gathered arrays and truncate to the number of samples (since the sampler added some extras
        to get each process a dataset of the same length).
        """
        ...

@dataclass
class LabelSmoother:
    """
    Adds label-smoothing on a pre-computed output from a Transformers model.

    Args:
        epsilon (`float`, *optional*, defaults to 0.1):
            The label smoothing factor.
        ignore_index (`int`, *optional*, defaults to -100):
            The index in the labels to ignore when computing the loss.
    """

    epsilon: float = ...
    ignore_index: int = ...
    def __call__(self, model_output, labels, shift_labels=...): ...

def get_length_grouped_indices(lengths, batch_size, mega_batch_mult=..., generator=...):  # -> list[Any]:
    """
    Return a list of indices so that each slice of `batch_size` consecutive indices correspond to elements of similar
    lengths. To do this, the indices are:

    - randomly permuted
    - grouped in mega-batches of size `mega_batch_mult * batch_size`
    - sorted by length in each mega-batch

    The result is the concatenation of all mega-batches, with the batch of `batch_size` containing the element of
    maximum length placed first, so that an OOM happens sooner rather than later.
    """
    ...

class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """
    def __init__(
        self,
        batch_size: int,
        dataset: Dataset | None = ...,
        lengths: list[int] | None = ...,
        model_input_name: str | None = ...,
        generator=...,
    ) -> None: ...
    def __len__(self):  # -> int:
        ...
    def __iter__(self):  # -> Iterator[Any]:
        ...

class DistributedLengthGroupedSampler(DistributedSampler):
    r"""
    Distributed Sampler that samples indices in a way that groups together features of the dataset of roughly the same
    length while keeping a bit of randomness.
    """
    def __init__(
        self,
        batch_size: int,
        dataset: Dataset | None = ...,
        num_replicas: int | None = ...,
        rank: int | None = ...,
        seed: int = ...,
        drop_last: bool = ...,
        lengths: list[int] | None = ...,
        model_input_name: str | None = ...,
    ) -> None: ...
    def __iter__(self) -> Iterator: ...

class ShardSampler(Sampler):
    """
    Sampler that shards batches between several processes. Dispatches indices batch by batch: on 2 processes with batch
    size 4, the first two batches are `[0, 1, 2, 3, 4, 5, 6, 7]` and `[8, 9, 10, 11, 12, 13, 14, 15]`, which shard into
    `[0, 1, 2, 3]` and `[8, 9, 10, 11]` for GPU-0 and `[4, 5, 6, 7]` and `[12, 13, 14, 15]` for GPU-1.

    The sampler thus yields `[0, 1, 2, 3, 8, 9, 10, 11]` on GPU-0 and `[4, 5, 6, 7, 12, 13, 14, 15]` on GPU-1.
    """
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = ...,
        drop_last: bool = ...,
        num_processes: int = ...,
        process_index: int = ...,
    ) -> None: ...
    def __iter__(self):  # -> Iterator[Any]:
        ...
    def __len__(self):  # -> int:
        ...

class IterableDatasetShard(IterableDataset):
    """
    Wraps a PyTorch `IterableDataset` to generate samples for one of the processes only. Instances of this class will
    always yield a number of samples that is a round multiple of the actual batch size (which is `batch_size x
    num_processes`). Depending on the value of the `drop_last` attribute, it will either stop the iteration at the
    first batch that would be too small or loop with indices from the beginning.

    On two processes with an iterable dataset yielding of `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]` with a batch size of
    2:

    - the shard on process 0 will yield `[0, 1, 4, 5, 8, 9]` so will see batches `[0, 1]`, `[4, 5]`, `[8, 9]`
    - the shard on process 1 will yield `[2, 3, 6, 7, 10, 11]` so will see batches `[2, 3]`, `[6, 7]`, `[10, 11]`

    <Tip warning={true}>

        If your IterableDataset implements some randomization that needs to be applied the same way on all processes
        (for instance, a shuffling), you should use a `torch.Generator` in a `generator` attribute of the `dataset` to
        generate your random numbers and call the [`~trainer_pt_utils.IterableDatasetShard.set_epoch`] method of this
        object. It will set the seed of this `generator` to `seed + epoch` on all processes before starting the
        iteration. Alternatively, you can also implement a `set_epoch()` method in your iterable dataset to deal with
        this.

    </Tip>

    Args:
        dataset (`torch.utils.data.IterableDataset`):
            The batch sampler to split in several shards.
        batch_size (`int`, *optional*, defaults to 1):
            The size of the batches per shard.
        drop_last (`bool`, *optional*, defaults to `False`):
            Whether or not to drop the last incomplete batch or complete the last batches by using the samples from the
            beginning.
        num_processes (`int`, *optional*, defaults to 1):
            The number of processes running concurrently.
        process_index (`int`, *optional*, defaults to 0):
            The index of the current process.
        seed (`int`, *optional*, defaults to 0):
            A random seed that will be used for the random number generation in
            [`~trainer_pt_utils.IterableDatasetShard.set_epoch`].
    """
    def __init__(
        self,
        dataset: IterableDataset,
        batch_size: int = ...,
        drop_last: bool = ...,
        num_processes: int = ...,
        process_index: int = ...,
        seed: int = ...,
    ) -> None: ...
    def set_epoch(self, epoch):  # -> None:
        ...
    def __iter__(self):  # -> Generator[Any, Any, None]:
        ...
    def __len__(self):  # -> int:
        ...

def metrics_format(self, metrics: dict[str, float]) -> dict[str, float]:
    """
    Reformat Trainer metrics values to a human-readable format

    Args:
        metrics (`Dict[str, float]`):
            The metrics returned from train/evaluate/predict

    Returns:
        metrics (`Dict[str, float]`): The reformatted metrics
    """
    ...

def log_metrics(self, split, metrics):  # -> None:
    """
    Log metrics in a specially formatted way

    Under distributed environment this is done only for a process with rank 0.

    Args:
        split (`str`):
            Mode/split name: one of `train`, `eval`, `test`
        metrics (`Dict[str, float]`):
            The metrics returned from train/evaluate/predictmetrics: metrics dict

    Notes on memory reports:

    In order to get memory usage report you need to install `psutil`. You can do that with `pip install psutil`.

    Now when this method is run, you will see a report that will include: :

    ```
    init_mem_cpu_alloc_delta   =     1301MB
    init_mem_cpu_peaked_delta  =      154MB
    init_mem_gpu_alloc_delta   =      230MB
    init_mem_gpu_peaked_delta  =        0MB
    train_mem_cpu_alloc_delta  =     1345MB
    train_mem_cpu_peaked_delta =        0MB
    train_mem_gpu_alloc_delta  =      693MB
    train_mem_gpu_peaked_delta =        7MB
    ```

    **Understanding the reports:**

    - the first segment, e.g., `train__`, tells you which stage the metrics are for. Reports starting with `init_`
        will be added to the first stage that gets run. So that if only evaluation is run, the memory usage for the
        `__init__` will be reported along with the `eval_` metrics.
    - the third segment, is either `cpu` or `gpu`, tells you whether it's the general RAM or the gpu0 memory
        metric.
    - `*_alloc_delta` - is the difference in the used/allocated memory counter between the end and the start of the
        stage - it can be negative if a function released more memory than it allocated.
    - `*_peaked_delta` - is any extra memory that was consumed and then freed - relative to the current allocated
        memory counter - it is never negative. When you look at the metrics of any stage you add up `alloc_delta` +
        `peaked_delta` and you know how much memory was needed to complete that stage.

    The reporting happens only for process of rank 0 and gpu 0 (if there is a gpu). Typically this is enough since the
    main process does the bulk of work, but it could be not quite so if model parallel is used and then other GPUs may
    use a different amount of gpu memory. This is also not the same under DataParallel where gpu0 may require much more
    memory than the rest since it stores the gradient and optimizer states for all participating GPUS. Perhaps in the
    future these reports will evolve to measure those too.

    The CPU RAM metric measures RSS (Resident Set Size) includes both the memory which is unique to the process and the
    memory shared with other processes. It is important to note that it does not include swapped out memory, so the
    reports could be imprecise.

    The CPU peak memory is measured using a sampling thread. Due to python's GIL it may miss some of the peak memory if
    that thread didn't get a chance to run when the highest memory was used. Therefore this report can be less than
    reality. Using `tracemalloc` would have reported the exact peak memory, but it doesn't report memory allocations
    outside of python. So if some C++ CUDA extension allocated its own memory it won't be reported. And therefore it
    was dropped in favor of the memory sampling approach, which reads the current process memory usage.

    The GPU allocated and peak memory reporting is done with `torch.cuda.memory_allocated()` and
    `torch.cuda.max_memory_allocated()`. This metric reports only "deltas" for pytorch-specific allocations, as
    `torch.cuda` memory management system doesn't track any memory allocated outside of pytorch. For example, the very
    first cuda call typically loads CUDA kernels, which may take from 0.5 to 2GB of GPU memory.

    Note that this tracker doesn't account for memory allocations outside of [`Trainer`]'s `__init__`, `train`,
    `evaluate` and `predict` calls.

    Because `evaluation` calls may happen during `train`, we can't handle nested invocations because
    `torch.cuda.max_memory_allocated` is a single counter, so if it gets reset by a nested eval call, `train`'s tracker
    will report incorrect info. If this [pytorch issue](https://github.com/pytorch/pytorch/issues/16266) gets resolved
    it will be possible to change this class to be re-entrant. Until then we will only track the outer level of
    `train`, `evaluate` and `predict` methods. Which means that if `eval` is called during `train`, it's the latter
    that will account for its memory usage and that of the former.

    This also means that if any other tool that is used along the [`Trainer`] calls
    `torch.cuda.reset_peak_memory_stats`, the gpu peak memory stats could be invalid. And the [`Trainer`] will disrupt
    the normal behavior of any such tools that rely on calling `torch.cuda.reset_peak_memory_stats` themselves.

    For best performance you may want to consider turning the memory profiling off for production runs.
    """
    ...

def save_metrics(self, split, metrics, combined=...):  # -> None:
    """
    Save metrics into a json file for that split, e.g. `train_results.json`.

    Under distributed environment this is done only for a process with rank 0.

    Args:
        split (`str`):
            Mode/split name: one of `train`, `eval`, `test`, `all`
        metrics (`Dict[str, float]`):
            The metrics returned from train/evaluate/predict
        combined (`bool`, *optional*, defaults to `True`):
            Creates combined metrics by updating `all_results.json` with metrics of this call

    To understand the metrics please read the docstring of [`~Trainer.log_metrics`]. The only difference is that raw
    unformatted numbers are saved in the current method.

    """
    ...

def save_state(self):  # -> None:
    """
    Saves the Trainer state, since Trainer.save_model saves only the tokenizer with the model

    Under distributed environment this is done only for a process with rank 0.
    """
    ...

def get_model_param_count(model, trainable_only=...):  # -> int:
    """
    Calculate model's total param count. If trainable_only is True then count only those requiring grads
    """
    ...

def get_parameter_names(model, forbidden_layer_types, forbidden_layer_names=...):  # -> list[Any]:
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    ...

def get_module_class_from_name(module, name):  # -> None:
    """
    Gets a class from a module by its name.

    Args:
        module (`torch.nn.Module`): The module to get the class from.
        name (`str`): The name of the class.
    """
    ...

def remove_dummy_checkpoint(is_main_process, output_dir, filenames):  # -> None:
    ...

if is_sagemaker_mp_enabled():
    @smp.step()
    def smp_forward_backward(model, inputs, gradient_accumulation_steps=...): ...
    @smp.step()
    def smp_forward_only(model, inputs): ...
    def smp_gather(tensor):  # -> list[Any] | tuple[Any, ...] | dict[Any, Any] | Tensor:
        ...
    def smp_nested_concat(tensor):  # -> list[Any] | tuple[Any, ...] | dict[Any, Any]:
        ...

@dataclass
class AcceleratorConfig:
    """
    A subset of arguments relating to the underlying [`accelerate.Accelerator`]
    implementation utilized in the `Trainer` that can be customized.
    Mostly relating to data.

    Parameters:
        split_batches (`bool`, *optional*, defaults to `False`):
            Whether or not the accelerator should split the batches yielded by the dataloaders across the devices. If
            `True` the actual batch size used will be the same on any kind of distributed processes, but it must be a
            round multiple of the `num_processes` you are using. If `False`, actual batch size used will be the one set
            in your script multiplied by the number of processes.
        dispatch_batches (`bool`, *optional*):
            If set to `True`, the dataloader prepared by the Accelerator is only iterated through on the main process
            and then the batches are split and broadcast to each process. Will default to `True` for `DataLoader` whose
            underlying dataset is an `IterableDataset`, `False` otherwise.
        even_batches (`bool`, *optional*, defaults to `True`):
            If set to `True`, in cases where the total batch size across all processes does not exactly divide the
            dataset, samples at the start of the dataset will be duplicated so the batch can be divided equally among
            all workers.
        use_seedable_sampler (`bool`, *optional*, defaults to `True`):
            Whether or not use a fully seedable random sampler ([`accelerate.data_loader.SeedableRandomSampler`]). Ensures
            training results are fully reproducible using a different sampling technique. While seed-to-seed results
            may differ, on average the differences are negligible when using multiple different seeds to compare. Should
            also be ran with [`~utils.set_seed`] for the best results.
        gradient_accumulation_kwargs (`dict`, *optional*):
            Additional kwargs to configure gradient accumulation, see [`accelerate.utils.GradientAccumulationPlugin`].
            Any of the following (optional) keys are acceptable:
              num_steps (`int`): Will take precedence over [`~.TrainingArguments.gradient_accumulation_steps`] if
                the latter is set to 1, otherwise an exception will be raised.
              adjust_scheduler (`bool`): Whether to adjust the scheduler steps to account for [`~.TrainingArguments.gradient_accumulation_steps`].
                The [`accelerate.utils.GradientAccumulationPlugin`] default is `True`.
              sync_each_batch (`bool`): Whether to synchronize the gradients at each data batch.
                The [`accelerate.utils.GradientAccumulationPlugin`] default is `False`.
        non_blocking (`bool`, *optional*, defaults to `False`):
            Whether to use non-blocking CUDA calls to help minimize synchronization during
            distributed training with prepared `DataLoader` inputs being moved to device.
            Best if used with `pin_memory=True` in the `TrainingArguments`.
        use_configured_state (`bool*, *optional*, defaults to `False`):
            Whether or not to use a pre-configured `AcceleratorState` or `PartialState` defined
            before calling `TrainingArguments`. If `True`, an `Accelerator` or `PartialState`
            must be initialized. May lead to issues using sweeps or hyperparameter tuning.

    """

    split_batches: bool = ...
    dispatch_batches: bool = ...
    even_batches: bool = ...
    use_seedable_sampler: bool = ...
    non_blocking: bool | None = ...
    gradient_accumulation_kwargs: dict | None = ...
    use_configured_state: bool = ...
    @classmethod
    def from_json_file(cls, json_file):  # -> Self:
        ...
    def to_dict(self):  # -> dict[str, Any]:
        ...
    def pop(self, key, default=...):  # -> Any:
        ...

class LayerWiseDummyOptimizer(torch.optim.Optimizer):
    """
    For Layer-wise optimizers such as GaLoRE optimizer, the optimization
    step is already done through the post gradient hooks. Therefore
    the trick is to create a dummy optimizer that can take arbitrary
    args and kwargs and return a no-op during training.

    Initial idea from @hiyouga in LLaMA-Factory:
    https://github.com/hiyouga/LLaMA-Factory/commit/8664262cde3919e10eaecbd66e8c5d356856362e#diff-ebe08ab14496dfb9e06075f0fdd36799ef6d1535cc4dd4715b74c4e3e06fe3ba
    """
    def __init__(self, optimizer_dict=..., *args, **kwargs) -> None: ...
    def zero_grad(self, set_to_none: bool = ...) -> None: ...
    def step(self, closure=...) -> float | None: ...

class LayerWiseDummyScheduler(LRScheduler):
    """
    For Layer-wise optimizers such as GaLoRE optimizer, the optimization and scheduling step
    are already done through the post gradient hooks. Therefore
    the trick is to create a dummy scheduler that can take arbitrary
    args and kwargs and return a no-op during training.
    """
    def __init__(self, *args, **kwargs) -> None: ...
    def get_lr(self):  # -> list[Any]:
        ...

def set_rng_state_for_device(device_name, device_module, checkpoint_rng_state, is_distributed):  # -> None:
    """Helper to set RNG state for a specific device type (CUDA, NPU, MLU, MUSA)"""
    ...
