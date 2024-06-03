#! /usr/bin/env python3


from pathlib import Path
from ruamel.yaml import YAML
from typing import TypeVar, Generic, Generator, Union, Dict, Optional


T = TypeVar("T")


class YamlDatabase(Generic[T]):
    def __init__(self, path: Union[Path, str], contained_type: Optional[type] = None) -> None:
        self.path = Path(path).expanduser()
        self.path.touch(exist_ok=True)

        self.yaml = YAML()
        self.yaml.indent(mapping=2, sequence=4, offset=2)
        self.yaml.explicit_start = True
        self.yaml.explicit_end = True
        if contained_type is not None:
            self.yaml.register_class(contained_type)

        self.data = self._load()

    def __getitem__(self, __name: str) -> T:
        return self.data[__name]

    def __setitem__(self, __name: str, __value: T) -> None:
        self.data[__name] = __value

    def __delitem__(self, __name: str) -> None:
        if __name in self.data:
            del self.data[__name]

    def __contains__(self, __name: str) -> bool:
        return __name in self.data.keys()

    def get(self, name: str) -> T:
        if name not in self.data.keys():
            raise IndexError(f"{name} not found in database")
        else:
            return self.data[name]

    def remove(self, name: str) -> None:
        if name not in self.data.keys():
            raise IndexError(f"{name} not found in database")
        else:
            del self.data[name]

    def add(self, name: str, element: T) -> None:
        if name in self.data.keys():
            raise IndexError(f"{name} already exists in database")
        else:
            self.data[name] = element

    def rename(self, old_name: str, new_name: str) -> None:
        if old_name not in self.data.keys():
            raise IndexError(f"{old_name} not found in database")
        elif new_name in self.data.keys():
            raise IndexError(f"{new_name} already exists in database")
        else:
            self.data[new_name] = self.data.pop(old_name)

    def replace(self, name: str, value: T) -> None:
        if name not in self.data.keys():
            raise IndexError(f"{name} not found in database")
        else:
            self.data[name] = value

    def commit(self) -> None:
        self.yaml.dump(self.data, self.path)
        self.refresh()

    def refresh(self) -> None:
        self.data = self._load()

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)
        self.path.touch()
        self.refresh()

    def values(self) -> Generator[T, None, None]:
        return (v for v in self.data.values())

    def _load(self) -> Dict[str, T]:
        return self.yaml.load(self.path) or {}
