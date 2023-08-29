from pathlib import Path


def to_path(path):
    if isinstance(path, Path):
        return path
    else:
        return Path(path)
