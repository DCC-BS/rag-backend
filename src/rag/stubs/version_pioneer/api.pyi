from pathlib import Path

from version_pioneer.versionscript import VersionDict

def get_version_dict_wo_exec(
    cwd: Path,
    style: str,
    tag_prefix: str,
) -> VersionDict: ...
