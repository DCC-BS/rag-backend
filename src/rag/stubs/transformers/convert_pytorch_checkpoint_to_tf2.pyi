"""
This type stub file was generated by pyright.
"""

from . import is_torch_available

"""Convert pytorch checkpoints to TensorFlow"""
if is_torch_available(): ...
MODEL_CLASSES = ...

def convert_pt_checkpoint_to_tf(
    model_type, pytorch_checkpoint_path, config_file, tf_dump_path, compare_with_pt_model=..., use_cached_models=...
):  # -> None:
    ...
def convert_all_pt_checkpoints_to_tf(
    args_model_type,
    tf_dump_path,
    model_shortcut_names_or_path=...,
    config_shortcut_names_or_path=...,
    compare_with_pt_model=...,
    use_cached_models=...,
    remove_cached_files=...,
    only_convert_finetuned_models=...,
):  # -> None:
    ...

if __name__ == "__main__":
    parser = ...
    args = ...
