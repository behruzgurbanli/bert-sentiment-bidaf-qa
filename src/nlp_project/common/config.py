#!/usr/bin/env python3
"""
Minimal config helpers shared by CLI and UI.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Type, TypeVar, get_args, get_origin

import yaml

T = TypeVar("T")


def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p.resolve()}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping. Got: {type(data)}")
    return data


def _is_path_type(t: Any) -> bool:
    if t is Path:
        return True
    origin = get_origin(t)
    if origin is not None:
        return any(_is_path_type(arg) for arg in get_args(t))
    if isinstance(t, str):
        return "Path" in t
    return False


def load_config_as(path: str | Path, cls: Type[T]) -> T:
    if not is_dataclass(cls):
        raise TypeError("load_config_as expects a dataclass type")

    data = load_yaml(path)
    allowed = {f.name: f for f in fields(cls)}
    kwargs: Dict[str, Any] = {}
    for k, v in data.items():
        if k not in allowed:
            continue
        field_info = allowed[k]
        if _is_path_type(field_info.type) and isinstance(v, (str, Path)):
            kwargs[k] = Path(v)
        else:
            kwargs[k] = v
    return cls(**kwargs)  # type: ignore[arg-type]
