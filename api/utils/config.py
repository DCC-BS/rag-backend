from pathlib import Path

from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig

_config = None


def load_config() -> DictConfig | ListConfig:
    """Loads and merges configuration files using OmegaConf."""
    global _config
    if _config is None:
        OmegaConf.clear_resolvers()
        _config = OmegaConf.create()
        cli_conf = OmegaConf.from_cli()
        _config = OmegaConf.merge(_config, cli_conf)

        # Recursively find all yaml files in conf directory
        conf_dir = Path("conf")
        yaml_files = sorted(
            [f for f in conf_dir.rglob("*.yaml") if f.is_file()], key=lambda x: x.as_posix()
        )

        for conf_file in yaml_files:
            try:
                file_conf = OmegaConf.load(file_=conf_file)
                _config = OmegaConf.merge(_config, file_conf)
            except Exception as e:
                print(f"Error loading config from {conf_file}: {e}")
    return _config


def get_config() -> DictConfig | ListConfig:
    """
    Retrieves the loaded configuration.

    Returns:
        omegaconf.DictConfig: The loaded configuration.
    """
    if _config is None:
        raise ValueError("Configuration has not been loaded. Call load_config() first.")
    return _config
