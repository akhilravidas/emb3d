"""
Job Execution
"""
import asyncio
import threading

from rich import print
from rich.console import Console
from rich.live import Live

from emb3d import textui
from emb3d.job import local, remote
from emb3d.types import EmbedJob


def execute(job: EmbedJob):
    console = Console()
    console.rule("Starting Job")
    with Live(auto_refresh=True, console=console) as live:
        print("Job Config:")
        print(job.describe())
        if job.execution_config.is_remote:
            asyncio.run(remote.run(job, textui.render_ui_async(job, live)))
        else:
            ui_thread = threading.Thread(target=textui.render_ui_sync, args=(job, live))
            ui_thread.start()
            local.run(job)
            ui_thread.join()
    console.rule("Job Complete")
