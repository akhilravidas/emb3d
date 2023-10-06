import io
import json
from unittest.mock import Mock, patch

import numpy as np
import pytest

from emb3d import reader
from emb3d.job.local import run
from emb3d.test_utils import mock_embed_job


# Mock for gen_batch, returns two mock batches
def mock_gen_batch(*args, **kwargs):
    mock_batch1 = Mock()
    mock_batch1.inputs = ["Hello world!"]

    mock_batch2 = Mock()
    mock_batch2.inputs = ["Testing 123"]

    return [mock_batch1, mock_batch2]


# Mock for config.max_tokens, returns a mock value
def mock_max_tokens(*args, **kwargs):
    return 1000


@pytest.mark.parametrize(
    "inputs, expected_embeddings",
    [
        (["Hello world!", "Bye world!"], np.array([[0.1, 0.2], [0.5, 0.6]])),
        (["Testing 123"], np.array([[0.7, 0.8]])),
    ],
)
def test_run(inputs, expected_embeddings):
    mock_model = Mock()
    mock_model.encode.return_value = expected_embeddings

    with patch(
        "sentence_transformers.SentenceTransformer", return_value=mock_model
    ) as mock_transformer:
        records = [{"text": input} for input in inputs]
        in_file = io.StringIO("\n".join(json.dumps(record) for record in records))
        job = mock_embed_job(in_file=in_file, batch_size=100)
        run(job)

        mock_transformer.assert_called_once_with(job.model_id)
        mock_model.encode.assert_called_with(inputs)
        out_file = job.out_file
        out_file.seek(0)
        saved_records = list(reader.jsonl(out_file))
        print(saved_records)
        assert len(saved_records) == len(inputs)
        for idx, _ in enumerate(inputs):
            assert saved_records[idx]["embedding"] == expected_embeddings[idx].tolist()
