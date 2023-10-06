import io
import json
from unittest.mock import Mock, patch

import numpy as np
import pytest

from emb3d import reader
from emb3d.compute.local import run
from emb3d.test_utils import mock_embed_job


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
