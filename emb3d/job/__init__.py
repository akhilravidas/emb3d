"""
Job Execution
"""
import asyncio

from emb3d.job import local, remote
from emb3d.types import EmbedJob


def execute(job: EmbedJob):
    asyncio.run(remote.run(job))
