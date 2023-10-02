import asyncio
import json
from typing import Iterable, Iterator, Set

from aiolimiter import AsyncLimiter
from rich.live import Live

import textui

import client
import config
import reader
from etypes import Batch, EmbedJob, Failure, Result, WaitFor


def gen_batch(job: EmbedJob, batch_size: int, max_tokens: int) -> Iterator[Batch]:
    """
    Generates batches of rows from the input file.

    Batches are constructed so that:
    - Each batch contains atmost `batch_size` rows.
    - Each batch have atmost max_tokens (except when a single line exceeds token limit)
    """
    batch_ids = []
    batch_inputs = []
    batch_token_count = 0
    for line_num, record in enumerate(reader.jsonl(job.in_file)):
        text = record[job.column_name]
        new_tokens = client.approx_token_count(job, text)

        can_merge_token = batch_token_count + new_tokens < max_tokens or len(batch_ids) == 0
        can_merge_element = len(batch_ids) + 1 < batch_size

        can_merge = can_merge_token and can_merge_element
        if can_merge:
            batch_ids.append(line_num)
            batch_inputs.append(text)
            batch_token_count += new_tokens
        else:
            yield Batch(batch_ids, batch_inputs)
            batch_ids = [line_num]
            batch_inputs = [text]
            batch_token_count = new_tokens

    if batch_ids:
        yield Batch(batch_ids, batch_inputs)


# TODO: Implement insta fails [empty string]
async def generate(run: EmbedJob, rate_limiter: AsyncLimiter, batch: Batch, queue: asyncio.Queue, num_retries=2):
    assert len(batch.inputs) == len(batch.row_ids)
    while num_retries > 0:
        await rate_limiter.acquire()
        resp = await client.gen(run, batch.inputs)
        num_retries -= 1
        match resp:
            case Result(data):
                assert len(data) == len(batch.inputs)
                batch.embeddings = data
                run.batch_success(len(batch.inputs))
                queue.put_nowait(batch)
                break
            case Failure(error):
                batch.error = error
            case WaitFor(seconds, error):
                batch.error = error
                if num_retries != 0:
                    await asyncio.sleep(seconds)

        if num_retries == 0:
            run.batch_failure(len(batch.inputs))
            queue.put_nowait(batch)


async def flush(run: EmbedJob, queue: asyncio.Queue):
    """
    Writer task that flushes computed results to disk.
    """
    with open(run.out_file, "a") as fp:
        while True:
            result = await queue.get()
            if result.error is not None:
                run.batch_error(str(result.error))
            for idx, _ in enumerate(result.row_ids):
                fp.write(json.dumps({
                    "row_id": result.row_ids[idx],
                    "input": result.inputs[idx],
                    "embedding": result.embeddings[idx] if result.embeddings else None,
                    "error": str(result.error) if result.error else None,
                }) + "\n")
                run.batch_saved(1)
            queue.task_done()


async def cancel(tasks: Iterable[asyncio.Task]):
    for task in tasks:
        task.cancel()
    for task in tasks:
        try:
            await task
        except asyncio.CancelledError:
            pass

async def run(run: EmbedJob):
    results_q = asyncio.Queue()
    writer_handle = asyncio.create_task(flush(run, results_q))
    request_limiter = AsyncLimiter(config.max_requests_per_minute(run.backend), 60)
    in_progress_tasks: Set[asyncio.Task] = set()
    with Live(refresh_per_second=10) as live:
        ui_task = asyncio.create_task(textui.update_ui(run.tracker, live))
        try:
            for batch in gen_batch(run, config.batch_size(run.backend), config.max_tokens(run.backend)):
                task = asyncio.create_task(generate(run, request_limiter, batch, results_q))
                in_progress_tasks.add(task)
                task.add_done_callback(in_progress_tasks.remove)
        except KeyboardInterrupt:
            print(f"Keyboard interrupt, saving {results_q.qsize()} items before terminating.")
            await cancel(in_progress_tasks)
        finally:
            for task in list(in_progress_tasks):
                await task
            await results_q.join()
            await cancel([writer_handle, ui_task])
            await client.cleanup()
