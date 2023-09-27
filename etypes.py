from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List


class Backend(Enum):
    OPENAI = "OpenAI"
    COHERE = "Cohere"
    HUGGINGFACE = "Hugging Face"


@dataclass
class Result:
    data: List[List[float]]


@dataclass
class Failure:
    error: str


@dataclass
class WaitFor:
    seconds: float
    error: str | None


EmbedResponse = Result | Failure | WaitFor

OpenAIModels = ("text-embedding-ada-002", )
CohereModels = ("embed-english-v2.0", "embed-english-light-v2.0", "embed-multilingual-v2.0")

@dataclass
class EmbedJob:
    in_file: Path
    out_file: Path
    model_id: str
    api_key: str
    column_name: str = "text"

    @property
    def backend(self) -> Backend:
        return self.backend_from_model(self.model_id)

    @classmethod
    def backend_from_model(cls, model_id: str) -> Backend:
        if model_id in OpenAIModels:
            return Backend.OPENAI
        elif model_id in CohereModels:
            return Backend.COHERE
        else:
            return Backend.HUGGINGFACE


@dataclass
class Batch:
    row_ids: List[int]
    inputs: List[str]
    embeddings: List[List[float]] | None = None
    error: str | None = None
