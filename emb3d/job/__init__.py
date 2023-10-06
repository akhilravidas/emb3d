"""
Job Execution
"""
import asyncio

from emb3d.job import local, remote
from emb3d.types import EmbedJob


def execute(job: EmbedJob):
    if job.execution_config.is_remote:
        asyncio.run(remote.run(job))
    else:
        local.run(job)
