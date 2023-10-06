import io

from emb3d.types import EmbedJob, ExecutionConfig


def mock_embed_job(**kwargs):
    defaults = {
        "job_id": "test",
        "in_file": io.StringIO(),
        "model_id": "model",
        "out_file": io.StringIO(""),
        "total_records": 10,
        "batch_size": 10,
        "max_concurrent_requests": 1,
        "execution_config": ExecutionConfig.local(),
    }

    defaults.update(kwargs)
    return EmbedJob(**defaults)
