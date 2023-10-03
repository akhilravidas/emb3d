"""
Type declarations
"""
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union


class Backend(Enum):
    """ Supported model backends """
    OPENAI = "OpenAI"
    COHERE = "Cohere"
    HUGGINGFACE = "Hugging Face"


@dataclass
class Result:
    """ Embedding call result wrapper for a batch """
    data: List[List[float]]


@dataclass
class Failure:
    """ Embedding call failure wrapper """
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

OpenAIModels = ("text-embedding-ada-002", )
CohereModels = ("embed-english-v2.0", "embed-english-light-v2.0", "embed-multilingual-v2.0")

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
class EmbedJob:
    """
    Single embedding job
    """
    job_id: str
    in_file: Path
    out_file: Path
    model_id: str
    api_key: str
    total_records: int
    max_workers: int
    column_name: str = "text"


    tracker: JobTracker = field(init=False)

    def __post_init__(self):
        self.tracker = JobTracker(
            job_id=self.job_id,
            total=self.total_records
        )

    def batch_success(self, cnt: int):
        """ Success callback """
        self.tracker.success += cnt

    def batch_failure(self, cnt: int):
        """ Failure callback """
        self.tracker.failed += cnt

    def batch_saved(self, cnt: int):
        """ Saved callback """
        self.tracker.saved += cnt

    def batch_error(self, error_msg: str):
        """ Error callback """
        self.tracker.recent_errors.append(error_msg)


    @property
    def backend(self) -> Backend:
        """ Model Backend """
        return self.backend_from_model(self.model_id)

    @classmethod
    def backend_from_model(cls, model_id: str) -> Backend:
        """ model_id -> backend enum """
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
