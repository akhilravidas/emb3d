"""
Rich UI widgets for job progress reporting
"""
import asyncio

from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, ProgressColumn, TextColumn
from rich.rule import Rule
from rich.table import Table, box
from rich.text import Text

from etypes import EmbedJob, JobTracker


class TaskProgressColumn(ProgressColumn):
    def __init__(self):
        super().__init__()

    def render(self, task) -> Text:
        max_width = len(str(task.total))
        completed_str = str(task.completed).rjust(max_width)
        total_str = str(task.total).rjust(max_width)
        return Text(f"{completed_str} / {total_str}")





class ProgressBar(Progress):
    def get_renderables(self):
        yield Rule("Records")
        overall_tasks = [task for task in self.tasks if task.fields.get("progress_type") == "overall"]
        run_tasks = [task for task in self.tasks if task.fields.get("progress_type") != "overall"]
        # Show overall progress at the end
        self.columns = (
            "{task.description}",
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        )
        yield self.make_tasks_table(run_tasks)
        self.columns = self.get_default_columns()
        yield Text("\n")
        yield Rule("Overall")
        yield self.make_tasks_table(overall_tasks)

    def update_values(self, tracker: JobTracker):
        for task in self.tasks:
            if task.fields.get("task_handle") == "processing":
                task.completed = tracker.encoding
            if task.fields.get("task_handle") == "failed":
                task.completed = tracker.failed
            if task.fields.get("task_handle") == "success":
                task.completed = tracker.saved
            if task.fields.get("task_handle") == "overall":
                # do the difference to get ETA
                self.advance(task.id, tracker.saved - task.completed)
        self.refresh()


async def render_ui(job: EmbedJob, live: Live):
    tracker = job.tracker
    progress = ProgressBar()
    progress.add_task("[magenta]Running", total=tracker.total, progress_type="other", task_handle="processing")
    progress.add_task("[red]Failed", total=tracker.total, progress_type="other", task_handle="failed")
    progress.add_task("[green]Saved", total=tracker.total, progress_type="other", task_handle="success")
    progress.add_task("Progress", total=tracker.total, progress_type="overall", task_handle="overall")
    # TODO: Handle clean termination and keyboard interrupt
    while tracker.saved < tracker.total:
        live.update(render_loop(tracker, progress))

        await asyncio.sleep(0.8)
    live.update(render_loop(tracker, progress))

def render_loop(tracker: JobTracker, progress: ProgressBar):
    table = Table.grid(expand=True)
    error_table = Table(show_header=False, box=box.SIMPLE, expand=True)
    for error in tracker.recent_errors:
        error_table.add_row("- " + str(error))

    progress.update_values(tracker)
    if tracker.recent_errors:
        table.add_row(Panel(error_table, title="[b] Recent Errors", title_align="left", border_style="red", padding=(1, 2), expand=True))
    table.add_row(Panel.fit(progress, title="[b]Jobs", title_align="left", border_style="green", padding=(1, 2)))

    return table