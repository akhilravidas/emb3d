"""
Job Execution
"""
import asyncio
import threading
from pathlib import Path
from typing import Optional

import altair as alt
import pandas as pd
from rich import print
from rich.console import Console
from rich.live import Live

from emb3d import textui
from emb3d.compute import local, remote, visualize
from emb3d.types import EmbedJob, VisualizeDisplayMode


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


def generate_html(
    embedding_file: Path,
    min_cluster_size: int,
    title_field: Optional[str],
    display_mode: VisualizeDisplayMode,
):
    with textui.SimpleProgressBar("Reading Data"):
        X, titles = visualize.get_data(embedding_file, title_field)

    n_records, n_dims = X.shape
    # Run dimensionality reduction
    with textui.SimpleProgressBar(
        f"Mapping {n_records} records from {n_dims}-dimensional space to 2D (using: UMAP)."
    ):
        X_reduced = visualize.umap_reduce(X)

    hdbscan_model = None
    if min_cluster_size > 0:
        with textui.SimpleProgressBar(
            f"Clustering with min_cluster_size {min_cluster_size} (using: HDSCAN)."
        ):
            hdbscan_model = visualize.run_hdbscan(X_reduced, min_cluster_size)

    with textui.SimpleProgressBar("Generating Visualization"):
        chart = visualize.generate_chart(X_reduced, titles, hdbscan_model)

        out_fname = embedding_file.with_suffix(".2d.html")

        with alt.data_transformers.enable("default"):
            chart.save(out_fname)
        return out_fname
