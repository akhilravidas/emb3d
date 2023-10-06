"""
App and Run Configuration parameters
"""
from __future__ import annotations

import functools
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional

import yaml

from emb3d.types import Backend

# Scale down factor
SCALE_DOWN_FACTOR = 0.8

max_requests_limits = {
    Backend.OPENAI: 10000,
    Backend.COHERE: 1500,  # Using trial key values
    Backend.HUGGINGFACE: 100,
}

RATE_LIMIT_WAIT_TIME_SECS = 0.5

max_token_limits = {
    Backend.OPENAI: 8191,
    Backend.COHERE: 8000,
    Backend.HUGGINGFACE: 512,
}


@functools.cache
def max_requests_per_minute(backend: Backend) -> int:
    base_rpm = max_requests_limits.get(backend, 100)
    return int(base_rpm * SCALE_DOWN_FACTOR)


def max_tokens(backend: Backend) -> int:
    return max_token_limits.get(backend, 512)


def app_data_root() -> Path:
    sys_data_root = Path.home() / ".cache"
    return Path(os.getenv("XDG_CACHE_HOME", sys_data_root)) / "emb3d"


def config_path() -> Path:
    return app_data_root() / "config.yaml"


@dataclass
class AppConfig:
    default_model: str = "text-embedding-ada-002"
    openai_token: Optional[str] = None
    cohere_token: Optional[str] = None
    huggingface_token: Optional[str] = None

    @classmethod
    def instance(cls) -> AppConfig:
        if hasattr(cls, "_config"):
            return cls._config

        if not config_path().exists():
            cls._config = cls()
            return cls._config
        else:
            with config_path().open() as f:
                cls._config = cls(**yaml.safe_load(f))
                return cls._config

    def save(self):
        if not os.path.exists(app_data_root()):
            os.makedirs(app_data_root())
        with config_path().open("w") as f:
            yaml.safe_dump(self.__dict__, f)

    def get(self, key: str) -> Optional[str]:
        valid_keys = {field.name for field in fields(AppConfig)}
        if key not in valid_keys:
            raise ValueError(f"'{key}' is not a valid configuration option")

        return getattr(self, key, None)

    def set(self, key: str, value: str):
        valid_keys = {field.name for field in fields(AppConfig)}
        if key not in valid_keys:
            raise ValueError(f"'{key}' is not a valid configuration option")

        setattr(self, key, value)
        self.save()
