"""
Type declarations
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, TextIO, Union


class Backend(Enum):
    """Supported model backends"""

    OPENAI = "OpenAI"
    COHERE = "Cohere"
    HUGGINGFACE = "Hugging Face"
    LOCAL = "Local Execution"


@dataclass
class Result:
    """Embedding call result wrapper for a batch"""

    data: List[List[float]]


@dataclass
class Failure:
    """Embedding call failure wrapper"""

    error: str


@dataclass
class WaitFor:
    """
    Embedding call rate limit response handler.
    Some services like hugging face provide a wait time before retrying.
    """

    seconds: float
    error: Optional[str]


EmbedResponse = Union[Result, Failure, WaitFor]

OpenAIModels = ("text-embedding-ada-002",)
CohereModels = (
    "embed-english-v2.0",
    "embed-english-light-v2.0",
    "embed-multilingual-v2.0",
)


@dataclass
class JobTracker:
    """
    Run stats for a single job
    """

    job_id: str
    success: int = 0
    encoding: int = 0
    failed: int = 0
    saved: int = 0
    total: int = 0
    recent_errors: deque = field(default_factory=lambda: deque(maxlen=5))


@dataclass
class ExecutionConfig:
    class ExecutionMode(Enum):
        REMOTE = "remote"
        LOCAL = "local"

    mode: ExecutionMode
    api_key: str

    @classmethod
    def local(cls) -> ExecutionConfig:
        return cls(cls.ExecutionMode.LOCAL, "")

    @classmethod
    def remote(cls, api_key: str) -> ExecutionConfig:
        return cls(cls.ExecutionMode.REMOTE, api_key)

    @property
    def is_remote(self) -> bool:
        return self.mode == self.ExecutionMode.REMOTE


@dataclass
class EmbedJob:
    """
    Single embedding job
    """

    job_id: str
    in_file: TextIO
    out_file: TextIO
    model_id: str
    total_records: int
    max_concurrent_requests: int
    execution_config: ExecutionConfig
    column_name: str = "text"
    tracker: JobTracker = field(init=False)

    def __post_init__(self):
        self.tracker = JobTracker(job_id=self.job_id, total=self.total_records)

    def batch_success(self, cnt: int):
        """Success callback"""
        self.tracker.success += cnt

    def batch_failure(self, cnt: int):
        """Failure callback"""
        self.tracker.failed += cnt

    def batch_saved(self, cnt: int):
        """Saved callback"""
        self.tracker.saved += cnt

    def batch_error(self, error_msg: str):
        """Error callback"""
        self.tracker.recent_errors.append(error_msg)

    @property
    def backend(self) -> Backend:
        """Model Backend"""
        return self.backend_from_model(self.model_id)

    @property
    def api_key(self) -> str:
        """API key"""
        return self.execution_config.api_key

    @classmethod
    def backend_from_model(cls, model_id: str) -> Backend:
        """model_id -> backend enum"""
        if model_id in OpenAIModels:
            return Backend.OPENAI
        elif model_id in CohereModels:
            return Backend.COHERE
        else:
            return Backend.HUGGINGFACE


@dataclass
class Batch:
    """
    Input batch
    """

    row_ids: List[int]
    inputs: List[str]
    embeddings: Optional[List[List[float]]] = None
    error: Optional[str] = None
