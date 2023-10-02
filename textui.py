# import asyncio

# from rich.table import Table, box
# from rich.panel import Panel
# from rich.progress import Progress
# from rich.live import Live

# async def update_ui(tracker: JobTracker, live: Live):
#     overall_progress = Progress(expand=True)
#     overall_task = overall_progress.add_task("Progress", total=tracker.total)
#     while True:
#         live.update(render_loop(tracker, overall_progress, overall_task))
#         await asyncio.sleep(0.2)

# def render_loop(tracker: JobTracker, progress: Progress, task_id: int):
#     table = Table(expand=True, title_justify="left", show_header=False, box=box.SIMPLE)
#     table.title = f"Job ID: {tracker.job_id}"
#     # There has to be a better way for ETA
#     progress.advance(task_id, tracker.saved - progress.tasks[task_id].completed)
#     table.add_row(progress)
#     return table

import asyncio
from rich.live import Live
from rich.table import Table, box
from rich.progress import Progress
from collections import deque
from dataclasses import dataclass, field
from etypes import JobTracker

async def update_ui(tracker: JobTracker, live: Live):
    overall_progress = Progress(expand=True)
    overall_task = overall_progress.add_task("Progress", total=tracker.total)

    while True:
        # Create tables for errors and progress
        error_table = Table(title="Errors", show_header=False, box=box.SIMPLE)
        progress_table = Table(title="Progress", show_header=False, box=box.SIMPLE)

        # Populate error table
        for error in tracker.recent_errors:
            error_table.add_row(str(error))

        # Populate progress table
        progress_table.add_row(f"Success: {tracker.success}")
        progress_table.add_row(f"Failed: {tracker.failed}")
        progress_table.add_row(f"Saved: {tracker.saved}")
        progress_table.add_row(f"Total: {tracker.total}")

        # Update live display
        live.update(render_loop(tracker, overall_progress, overall_task, error_table, progress_table))

        await asyncio.sleep(0.2)

def render_loop(tracker: JobTracker, progress: Progress, task_id: int, error_table: Table, progress_table: Table):
    table = Table(expand=True, title_justify="left", show_header=False, box=box.SIMPLE)
    table.title = f"Job ID: {tracker.job_id}"

    # Update progress
    progress.advance(task_id, tracker.saved - progress.tasks[task_id].completed)

    # # Add sub-tables to the main table
    table.add_row(progress_table)
    table.add_row(error_table)
    table.add_row(progress)

    return table
