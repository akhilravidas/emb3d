"""
Job Execution
"""
import asyncio
import threading

from rich.live import Live

from emb3d import textui
from emb3d.job import local, remote
from emb3d.types import EmbedJob


def execute(job: EmbedJob):
    with Live(auto_refresh=True) as live:
        live.console.rule("Embedding Job: " + job.job_id)
        if job.execution_config.is_remote:
            asyncio.run(remote.run(job, textui.render_ui_async(job, live)))
        else:
            ui_thread = threading.Thread(target=textui.render_ui_sync, args=(job, live))
            ui_thread.start()
            local.run(job)
            ui_thread.join()
