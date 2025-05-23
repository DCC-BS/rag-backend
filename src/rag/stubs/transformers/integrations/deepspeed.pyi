"""
This type stub file was generated by pyright.
"""

from ..utils import is_accelerate_available, is_torch_available

"""
Integration with Deepspeed
"""
if is_torch_available(): ...
logger = ...

def is_deepspeed_available():  # -> bool | None:
    ...

if is_accelerate_available() and is_deepspeed_available(): ...
else: ...

class HfDeepSpeedConfig:
    """
    This object contains a DeepSpeed configuration dictionary and can be quickly queried for things like zero stage.

    A `weakref` of this object is stored in the module's globals to be able to access the config from areas where
    things like the Trainer object is not available (e.g. `from_pretrained` and `_get_resized_embeddings`). Therefore
    it's important that this object remains alive while the program is still running.

    [`Trainer`] uses the `HfTrainerDeepSpeedConfig` subclass instead. That subclass has logic to sync the configuration
    with values of [`TrainingArguments`] by replacing special placeholder values: `"auto"`. Without this special logic
    the DeepSpeed configuration is not modified in any way.

    Args:
        config_file_or_dict (`Union[str, Dict]`): path to DeepSpeed config file or dict.

    """
    def __init__(self, config_file_or_dict) -> None: ...

class HfTrainerDeepSpeedConfig(HfDeepSpeedConfig):
    """
    The `HfTrainerDeepSpeedConfig` object is meant to be created during `TrainingArguments` object creation and has the
    same lifespan as the latter.
    """
    def __init__(self, config_file_or_dict) -> None: ...
    def dtype(self):  # -> dtype:
        ...
    def is_auto(self, ds_key_long):  # -> Literal[False]:
        ...
    def fill_match(self, ds_key_long, hf_val, hf_key=..., must_match=...):  # -> None:
        """
        A utility method that massages the config file and can optionally verify that the values match.

        1. Replace "auto" values with `TrainingArguments` value.

        2. If it wasn't "auto" and `must_match` is true, then check that DS config matches Trainer
        config values and if mismatched add the entry to `self.mismatched` - will assert during
        `trainer_config_finalize` for one or more mismatches.

        """
        ...

    fill_only = ...
    def trainer_config_process(self, args, auto_find_batch_size=...):  # -> None:
        """
        Adjust the config with `TrainingArguments` values. This stage is run during `TrainingArguments` object
        creation.
        """
        ...

    def trainer_config_finalize(self, args, model, num_training_steps):  # -> None:
        """
        This stage is run after we have the model and know num_training_steps.

        Now we can complete the configuration process.
        """
        ...

_hf_deepspeed_config_weak_ref = ...

def set_hf_deepspeed_config(hf_deepspeed_config_obj):  # -> None:
    ...
def unset_hf_deepspeed_config():  # -> None:
    ...
def is_deepspeed_zero3_enabled():  # -> Literal[False]:
    ...
def deepspeed_config():  # -> None:
    ...
def deepspeed_optim_sched(
    trainer, hf_deepspeed_config, args, num_training_steps, model_parameters
):  # -> tuple[Any, Any]:
    """
    A convenience wrapper that deals with optimizer and lr scheduler configuration.
    """
    ...

def deepspeed_init(trainer, num_training_steps, inference=...):  # -> tuple[Any | None, Any | None]:
    """
    Init DeepSpeed, after updating the DeepSpeed configuration with any relevant Trainer's args.

    If `resume_from_checkpoint` was passed then an attempt to resume from a previously saved checkpoint will be made.

    Args:
        trainer: Trainer object
        num_training_steps: per single gpu
        resume_from_checkpoint: path to a checkpoint if to resume from after normal DeepSpeedEngine load
        inference: launch in inference mode (no optimizer and no lr scheduler)
        auto_find_batch_size: whether to ignore the `train_micro_batch_size_per_gpu` argument as it's being
            set automatically by the auto batch size finder

    Returns: optimizer, lr_scheduler

    We may use `deepspeed_init` more than once during the life of Trainer, when we do - it's a temp hack based on:
    https://github.com/deepspeedai/DeepSpeed/issues/1394#issuecomment-937405374 until Deepspeed fixes a bug where it
    can't resume from a checkpoint after it did some stepping https://github.com/deepspeedai/DeepSpeed/issues/1612

    """
    ...

def deepspeed_load_checkpoint(deepspeed_engine, checkpoint_path, load_module_strict=...):  # -> None:
    ...
