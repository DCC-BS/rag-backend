from pathlib import Path

from version_pioneer.api import get_version_dict_wo_exec
from version_pioneer.versionscript import VersionDict


def get_version_dict() -> VersionDict:
    # NOTE: during installation, __file__ is not defined
    # When installed in editable mode, __file__ is defined
    # When installed in standard mode (when built), this file is replaced to a compiled versionfile.
    cwd = Path(__file__).parent if "__file__" in globals() else Path.cwd()

    # Cast the result to VersionDict to satisfy the type checker
    version_dict: VersionDict = get_version_dict_wo_exec(
        cwd=cwd,
        style="pep440",
        tag_prefix="v",
    )
    return version_dict
