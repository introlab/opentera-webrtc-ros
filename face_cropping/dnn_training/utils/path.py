from pathlib import Path


def to_path(path):
    return Path(path).expanduser()
