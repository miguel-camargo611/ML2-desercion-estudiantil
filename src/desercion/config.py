"""Carga de la configuración y resolución de rutas del proyecto."""

from __future__ import annotations

import os
import random
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "config.yaml"


@lru_cache(maxsize=1)
def load_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def path(key: str) -> Path:
    """Ruta absoluta para una entrada de ``paths`` en config.yaml; crea la carpeta si hace falta."""
    p = ROOT / load_config()["paths"][key]
    target_dir = p if p.suffix == "" else p.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    return p


def seed() -> int:
    return int(load_config()["seed"])


def set_global_seed(value: int | None = None) -> int:
    """Fija las semillas de Python y NumPy para garantizar repetibilidad."""
    value = seed() if value is None else value
    os.environ["PYTHONHASHSEED"] = str(value)
    random.seed(value)
    np.random.seed(value)
    return value
