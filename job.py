"""
Job Execution
"""
import asyncio
import json
from typing import Iterable, Iterator

from aiolimiter import AsyncLimiter
from rich.live import Live

import client
import config
import reader
import textui
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


async def terminate(*tasks: Iterable[asyncio.Task]):
    """ Cancels tasks and waits for them to complete. """
    for task in tasks:
        task.cancel()
    for task in tasks:
        try:
            await task
        except asyncio.CancelledError:
            pass

async def produce(job: EmbedJob, queue: asyncio.Queue):
    """
    Producer task that generates batches and pushes them to the queue.
    """
    for batch in gen_batch(job, config.batch_size(job.backend), config.max_tokens(job.backend)):
        await queue.put(batch)


FILE_WRITE_LOCK = asyncio.Lock()

async def write_batch_results(job: EmbedJob, batch: Batch):
    """ Write batch results to output file. """
    async with FILE_WRITE_LOCK:
        with open(job.out_file, "a") as fp:
            for idx, _ in enumerate(batch.row_ids):
                fp.write(json.dumps({
                    "row_id": batch.row_ids[idx],
                    "input": batch.inputs[idx],
                    "embedding": batch.embeddings[idx] if batch.embeddings else None,
                    "error": str(batch.error) if batch.error else None,
                }) + "\n")
    job.batch_saved(len(batch.row_ids))


async def worker(job: EmbedJob, rate_limiter: AsyncLimiter, job_queue: asyncio.Queue, num_retries=2):
    """
    Consumer task that consumes batches from the queue and generates embeddings.
    """
    while True:
        batch = await job_queue.get()
        assert len(batch.inputs) == len(batch.row_ids)
        batch_retry = num_retries
        while batch_retry > 0:
            batch_retry -= 1
            await rate_limiter.acquire()
            job.tracker.encoding += len(batch.inputs)
            resp = await client.gen(job, batch.inputs)
            job.tracker.encoding -= len(batch.inputs)
            # match resp:
            #     case Result(data):
            #         assert len(data) == len(batch.inputs)
            #         # clear off transient error
            #         batch.error = None
            #         batch.embeddings = data
            #         job.batch_success(len(batch.inputs))
            #         await write_batch_results(job, batch)
            #         break
            #     case Failure(error):
            #         batch.error = error
            #     case WaitFor(seconds, error):
            #         batch.error = error
            #         if batch_retry != 0:
            #             await asyncio.sleep(seconds)
            if isinstance(resp, Result):
                assert len(resp.data) == len(batch.inputs)
                # clear off transient error
                batch.error = None
                batch.embeddings = resp.data
                job.batch_success(len(batch.inputs))
                await write_batch_results(job, batch)
            elif isinstance(resp, Failure):
                batch.error = resp.error
            elif isinstance(resp, WaitFor):
                batch.error = resp.error
                if batch_retry != 0:
                    await asyncio.sleep(resp.seconds)

            if batch_retry == 0:
                job.batch_failure(len(batch.inputs))
                if batch.error is not None:
                    job.batch_error(str(batch.error))
                await write_batch_results(job, batch)
        job_queue.task_done()

async def consume(job: EmbedJob, rate_limiter: AsyncLimiter, job_queue: asyncio.Queue, num_retries=2):
    """
    Consumer task that consumes batches from the queue and generates embeddings.
    """
    return await asyncio.gather(*[worker(job, rate_limiter, job_queue, num_retries) for _ in range(job.max_workers)], return_exceptions=True)

async def run(job: EmbedJob):
    """
    Main entry point for the embedding job.
    """
    job_queue = asyncio.Queue(maxsize=job.max_workers)
    request_limiter = AsyncLimiter(config.max_requests_per_minute(job.backend), 60)
    with Live(auto_refresh=True) as live:
        live.console.rule("Embedding Job: " + job.job_id)
        ui_task = asyncio.create_task(textui.render_ui(job, live))
        producer_task = asyncio.create_task(produce(job, job_queue))
        consumer_task = asyncio.create_task(consume(job, request_limiter, job_queue=job_queue))
        try:
            await asyncio.wait({producer_task}, return_when=asyncio.FIRST_EXCEPTION)
            await job_queue.join()
            await terminate(consumer_task)
        except KeyboardInterrupt:
            print(f"Keyboard interrupt, saving {job_queue.qsize()} items before terminating.")
            await terminate(producer_task, consumer_task)
        finally:
            await client.cleanup()
            await ui_task