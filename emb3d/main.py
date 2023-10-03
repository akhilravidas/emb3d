"""
emb3d CLI
"""
import asyncio
import os
from pathlib import Path
from typing import Optional
from uuid import uuid4

import typer
from rich.prompt import Confirm, Prompt

from emb3d import reader
from emb3d.job import run
from emb3d.types import Backend, EmbedJob

app = typer.Typer()

default_env_variables = {
    Backend.OPENAI: "OPENAI_API_KEY",
    Backend.COHERE: "CO_API_KEY",
    Backend.HUGGINGFACE: "HUGGINGFACE_API_KEY"
}

@app.command(help="Generate embeddings for your input.")
def main(
    input_file: Optional[Path] = typer.Argument(None, help="Path to the input file. If not provided, it will be prompted."),
    model_id: Optional[str] = typer.Option(None, help="ID of the embedding model. Default is 'text-embedding-ada-002'."),
    output_file: Optional[Path] = typer.Option(None, "--output_file", "-o", help="Path to the output file. If not provided, a default path will be suggested."),
    api_key: Optional[str] = typer.Option(None, help="API key for the backend. If not provided, it will be prompted or fetched from environment variables."),
    max_concurrent_requests: int = typer.Option(1000, help="Maximum number of concurrent requests for the embedding task. Default is 1000."),
    yes: bool = typer.Option(False, "-y", "--yes", help="Confirm all prompts automatically.")
):
    if input_file is None:
        input_file = Path(Prompt.ask("Enter the input file path"))
        if not input_file.exists():
            raise typer.BadParameter(f"File {input_file} does not exist, aborting...")
        if not input_file.is_file():
            raise typer.BadParameter(f"File {input_file} is not a valid file, aborting...")
    if yes and model_id is None:
        model_id = "text-embedding-ada-002"
    else:
        model_id = Prompt.ask("Enter the embedding model", default="text-embedding-ada-002")
    if output_file is None:
        default_out_file = input_file.with_suffix(".embeddings.jsonl")
        idx = 0
        while default_out_file.exists():
            idx += 1
            default_out_file = input_file.with_suffix(f".run-{idx}.embeddings.jsonl")
        if yes or Confirm.ask(f"Save to [green]{default_out_file}[/green]?"):
            output_file = default_out_file
        else:
            output_file = Path(Prompt.ask(f"Enter the output file path"))
            if output_file.exists():
                raise typer.BadParameter(f"File {output_file} already exists, aborting...")

    backend = EmbedJob.backend_from_model(model_id)
    env_key = default_env_variables.get(backend)
    if env_key is not None:
        api_key = api_key or os.getenv(env_key)

    if api_key is None:
        if yes:
            raise typer.BadParameter(f"API key for {backend.value} backend is required, aborting...")
        else:
            api_key = Prompt.ask(f"Enter your [red][bold]{backend.value} API key[/bold][/red]").strip()

    num_records = sum(1 for _ in reader.line(input_file))

    job = EmbedJob(
        job_id=str(uuid4()),
        in_file=input_file,
        model_id=model_id,
        out_file=output_file,
        api_key=api_key,
        total_records=num_records,
        max_concurrent_requests=min(max_concurrent_requests, num_records)
    )
    asyncio.run(run(job))
