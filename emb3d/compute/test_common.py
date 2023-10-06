import io
import json

from emb3d.compute.common import gen_batch, write_batch_results_post_lock
from emb3d.test_utils import mock_embed_job
from emb3d.types import Batch


def test_write_batch_results_post_lock():
    job = mock_embed_job()
    batch = Batch(row_ids=[1], inputs=["hello"], embeddings=[[4, 2]], error=None)

    write_batch_results_post_lock(job, batch)

    job.out_file.seek(0)
    written_data = json.loads(job.out_file.read())

    assert written_data["row_id"] == 1
    assert written_data["input"] == "hello"
    assert written_data["embedding"] == [4, 2]
    assert written_data["error"] is None


def test_gen_batch():
    in_file = io.StringIO('{"text": "hello"}\n{"text": "world"}')
    job = mock_embed_job(in_file=in_file)

    batches = list(gen_batch(job, batch_size=1, max_tokens=5))

    assert len(batches) == 2
    assert batches[0].row_ids == [0]
    assert batches[0].inputs == ["hello"]
    assert batches[1].row_ids == [1]
    assert batches[1].inputs == ["world"]


def test_gen_batch_low_tokens():
    in_file = io.StringIO('{"text": "hello"}\n{"text": "world"}')
    job = mock_embed_job(in_file=in_file)

    batches = list(gen_batch(job, batch_size=1, max_tokens=1))

    # Each batch should have atleast one element when if its above the token limit
    assert len(batches) == 2
    assert batches[0].row_ids == [0]
    assert batches[0].inputs == ["hello"]
    assert batches[1].row_ids == [1]
    assert batches[1].inputs == ["world"]

    in_file.seek(0)
    batches = list(gen_batch(job, batch_size=10, max_tokens=1))

    # Each batch should have atleast one element when if its above the token limit
    assert len(batches) == 2
    assert batches[0].row_ids == [0]
    assert batches[0].inputs == ["hello"]
    assert batches[1].row_ids == [1]
    assert batches[1].inputs == ["world"]


def test_gen_batch_high_tokens():
    in_file = io.StringIO('{"text": "hello"}\n{"text": "world"}')
    job = mock_embed_job(in_file=in_file)

    batches = list(gen_batch(job, batch_size=1, max_tokens=100))

    assert len(batches) == 2
    assert batches[0].row_ids == [0]
    assert batches[0].inputs == ["hello"]
    assert batches[1].row_ids == [1]
    assert batches[1].inputs == ["world"]

    in_file.seek(0)
    batches = list(gen_batch(job, batch_size=10, max_tokens=10))

    assert len(batches) == 1
    assert batches[0].row_ids == [0, 1]
    assert batches[0].inputs == ["hello", "world"]
