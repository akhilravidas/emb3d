"""
emb3d CLI
"""
import os
import sys
from io import StringIO
from pathlib import Path
from typing import Optional, TextIO
from uuid import uuid4

import typer
from rich.prompt import Prompt
from typing_extensions import Annotated

from emb3d import job, reader
from emb3d.config import AppConfig
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
    default_model = AppConfig.instance().default_model
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
    config = AppConfig.instance()

    api_key = (
        api_key
        or config.get(config_keys[backend])
        or os.getenv(default_env_variables[backend])
    )

    if not api_key:
        raise typer.BadParameter(
            f"API key for {backend.value} backend is required, re-run the command with --api_key [your_key] or set {default_env_variables[backend]} environment variable."
        )

    return ExecutionConfig.remote(api_key=api_key)


@app.command(help="Get or set a configuration value.")
def config(
    key: str = typer.Argument(
        ...,
        help="The configuration key to set or get. ex: `default_model`, `openai_token` etc..",
    ),
    value: Optional[str] = typer.Argument(
        ...,
        help="The value associated with the key. If not provided, the current value will be printed.",
    ),
):
    config = AppConfig.instance()
    if value is None:
        current_value = config.get(key)
        if current_value is not None:
            typer.echo(f"{key} = {current_value}")
        else:
            typer.echo(f"{key} is not set.")
    else:
        config.set(key, value)
        typer.echo(f"Saved new value for {key} to emb3d config.")


@app.command(help="Compute embeddings for fun and profit.")
def compute(
    input_file: Optional[Path] = typer.Argument(
        ... if sys.stdin.isatty() else None,
        help="Path to the input file.",
    ),
    model: Optional[str] = typer.Option(
        AppConfig.instance().default_model,
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
            job_id=str(uuid4()),
            in_file=input_file_io,
            model_id=model,
            out_file=output_file_io,
            total_records=num_records,
            batch_size=batch_size,
            max_concurrent_requests=min(max_concurrent_requests, num_records),
            execution_config=execution_mode,
        )

        job.execute(new_job)
