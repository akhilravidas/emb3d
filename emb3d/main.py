"""
emb3d CLI
"""
import os
import random
import string
import sys
import webbrowser
from enum import Enum
from io import StringIO
from pathlib import Path
from typing import Optional, TextIO

import typer
from rich.prompt import Prompt
from typing_extensions import Annotated

from emb3d import compute, config, textui
from emb3d.compute import visualize
from emb3d.io import reader, writer
from emb3d.types import Backend, EmbedJob, ExecutionConfig

app = typer.Typer(add_completion=False)

# logging.basicConfig(level=logging.DEBUG)


def _input_file_or_stdin(input_file: Optional[Path], stdin_input: bool) -> TextIO:
    if stdin_input:
        # NOTE: This isn't memory efficient, stdin is primarily for convenience
        # Long jobs are better off using input_file flag
        return StringIO(sys.stdin.read())
    if input_file is None:
        input_file = Path(Prompt.ask("Enter the input file path"))
        if not input_file or not input_file.exists():
            raise typer.BadParameter(f"File {input_file} does not exist, aborting...")
    if not input_file.is_file():
        raise typer.BadParameter(f"File {input_file} not found, aborting...")
    return input_file.open()


def _output_file(
    out_file: Optional[Path], input_file: Optional[Path], stdin_input: bool
) -> TextIO:
    if out_file is not None:
        # TODO: Handle job termination/resume
        if out_file.exists():
            raise typer.BadParameter(f"File {out_file} already exists, aborting...")
        return out_file.open("w")
    elif stdin_input:
        return sys.stdout
    else:
        place_holder_suffix = input_file or Path("emb3d-run")
        default_out_file = place_holder_suffix.with_suffix(".out.jsonl")
        idx = 0
        while default_out_file.exists():
            typer.echo(
                f"Auto generating output file, file {default_out_file} already exists..."
            )
            idx += 1
            default_out_file = place_holder_suffix.with_suffix(f".{idx}.out.jsonl")
        return default_out_file.open("w")


def _pick_model(model_id: Optional[str]) -> str:
    default_model = config.AppConfig.instance().default_model
    return (
        model_id
        or default_model
        or Prompt.ask("Enter the embedding model", default="text-embedding-ada-002")
    )


def _execution_config(
    api_key: Optional[str], model_id: str, remote: bool
) -> ExecutionConfig:
    backend = EmbedJob.backend_from_model(model_id)
    remote_only_backends = (Backend.OPENAI, Backend.COHERE)
    if backend in remote_only_backends:
        remote = True

    if not remote:
        return ExecutionConfig.local()

    default_env_variables = {
        Backend.OPENAI: "OPENAI_API_KEY",
        Backend.COHERE: "CO_API_KEY",
        Backend.HUGGINGFACE: "HUGGINGFACE_API_KEY",
    }
    config_keys = {
        Backend.OPENAI: "openai_token",
        Backend.COHERE: "cohere_token",
        Backend.HUGGINGFACE: "huggingface_token",
    }
    cfg = config.AppConfig.instance()

    api_key = (
        api_key
        or cfg.get(config_keys[backend])
        or os.getenv(default_env_variables[backend])
    )

    if not api_key:
        raise typer.BadParameter(
            f"API key for {backend.value} backend is required, re-run the command with --api_key [your_key] or set {default_env_variables[backend]} environment variable."
        )

    return ExecutionConfig.remote(api_key=api_key)


@app.command("config", help="Get or set a configuration value.")
def cmd_config(
    key: str = typer.Argument(
        ...,
        help="The configuration key to set or get. ex: `default_model`, `openai_token` etc..",
    ),
    value: Optional[str] = typer.Argument(
        ...,
        help="The value associated with the key. If not provided, the current value will be printed.",
    ),
):
    cfg = config.AppConfig.instance()
    if value is None:
        current_value = cfg.get(key)
        if current_value is not None:
            typer.echo(f"{key} = {current_value}")
        else:
            typer.echo(f"{key} is not set.")
    else:
        cfg.set(key, value)
        typer.echo(f"Saved new value for {key} to emb3d config.")


def new_job_id(length: int = 5) -> str:
    """Generate a random string of the given length."""
    characters = string.hexdigits
    return "".join(random.choice(characters) for _ in range(length)).lower()


@app.command("compute", help="Compute embeddings for fun and profit.")
def cmd_compute(
    input_file: Optional[Path] = typer.Argument(
        ... if sys.stdin.isatty() else None,
        help="Path to the input file.",
    ),
    model: Optional[str] = typer.Option(
        config.AppConfig.instance().default_model,
        help="Embedding model to use.",
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output-file",
        "-o",
        help="Path to the output file. If not provided, a default path will be suggested.",
    ),
    api_key: Optional[str] = typer.Option(
        None,
        help="API key for the service hosting the model. If not provided, it will be prompted or fetched from environment variables.",
    ),
    remote: Annotated[
        bool,
        typer.Option(
            "--remote/--local",
            help="Choose whether to do inference locally or with an API token. This choice is available for sentence transformer and hugging face models. If a model cannot be run locally (ex: OpenAI models), this flag is ignored.",
        ),
    ] = True,
    batch_size: int = typer.Option(
        100,
        help="Batch size to use for inputs. Default is 100.",
    ),
    max_concurrent_requests: int = typer.Option(
        1000,
        help="(Remote Execution) Maximum number of concurrent requests for the embedding task. Default is 1000.",
    ),
):
    stdin_input = input_file is None

    input_file_io = _input_file_or_stdin(input_file, stdin_input)
    output_file_io = _output_file(output_file, input_file, stdin_input)
    model = _pick_model(model)
    execution_mode = _execution_config(api_key, model, remote)

    with input_file_io, output_file_io:
        num_records = sum(1 for _ in reader.line(input_file_io))
        # Rewind
        input_file_io.seek(0)
        new_job = EmbedJob(
            job_id=new_job_id(),
            in_file=input_file_io,
            model_id=model,
            out_file=output_file_io,
            total_records=num_records,
            batch_size=batch_size,
            max_concurrent_requests=min(max_concurrent_requests, num_records),
            execution_config=execution_mode,
        )

        compute.execute(new_job)


class ClusterOption(str, Enum):
    auto = "auto"
    cluster = "cluster"
    no_cluster = "no-cluster"


@app.command("visualize", help="Visualize generated embeddings.")
def cmd_visualize(
    embedding_file: Path = typer.Argument(
        ...,
        help="Path to the embedding file.",
    ),
    min_cluster_size: Optional[int] = typer.Option(
        None,
        help="Smallest size grouping that you wish to consider a cluster.",
    ),
    label_field: Optional[str] = typer.Option(
        "title",
        help="Field used to describe the record. If not provided, the `id` field (if present) or line numbers will be used.",
    ),
    cluster: Annotated[
        ClusterOption, typer.Option(case_sensitive=False)
    ] = ClusterOption.auto,
):
    with textui.SimpleProgressBar("Reading Data"):
        X, labels = visualize.get_data(embedding_file, label_field)

    n_records, n_dims = X.shape
    typer.echo(f"Loaded {n_records} records with {n_dims} dimensions.")
    with textui.SimpleProgressBar(
        f"Mapping records from {n_dims}-dimensional space to 2D (using: UMAP)."
    ):
        X_reduced = visualize.umap_reduce(X)

    needs_clustering = (
        cluster != ClusterOption.no_cluster
        and n_records > config.VISUALIZATION_CLUSTERING_THRESHOLD
    )

    if (
        n_records > config.VISUALIZATION_CLUSTERING_THRESHOLD
        and cluster != ClusterOption.cluster
    ):
        typer.echo(f"Too many records to visualize, clustering data...")

    min_cluster_size = min_cluster_size or config.VISUALIZATION_DEFAULT_MIN_CLUSTER_SIZE
    hdbscan_model = None
    if needs_clustering:
        with textui.SimpleProgressBar(
            f"Clustering with min_cluster_size {min_cluster_size} (using: HDSCAN)."
        ):
            hdbscan_model = visualize.cluster_hdbscan(X_reduced, min_cluster_size)

    with textui.SimpleProgressBar("Generating Scatter Plot"):
        chart = visualize.generate_chart(X_reduced, labels, hdbscan_model)
        out_file = embedding_file.with_suffix(".2d.html")
        writer.chart2html(chart, out_file)

    typer.echo(f"Visualization saved to {out_file}.")

    if typer.confirm("View in browser?"):
        webbrowser.open_new_tab("file://" + os.path.abspath(out_file))
