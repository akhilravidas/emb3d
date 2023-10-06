import asyncio
import logging

from aiolimiter import AsyncLimiter
from rich.live import Live

from emb3d import client, config, textui
from emb3d.job.common import gen_batch, write_batch_results_post_lock
from emb3d.types import Batch, EmbedJob, Failure, Result, WaitFor


async def terminate(*tasks):
    """Cancels tasks and waits for them to complete."""
    logging.debug("Terminating tasks...")
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
    logging.debug("Producer: Starting")
    for batch in gen_batch(
        job, config.batch_size(job.backend), config.max_tokens(job.backend)
    ):
        logging.debug("Producer: Next Batch [%d]", len(batch.row_ids))
        await queue.put(batch)


FILE_WRITE_LOCK = asyncio.Lock()


async def write_batch_results(job: EmbedJob, batch: Batch):
    """Write batch results to output file."""
    async with FILE_WRITE_LOCK:
        # We use synchronous file operations as the data being written is small
        write_batch_results_post_lock(job, batch)


async def worker(
    job: EmbedJob, rate_limiter: AsyncLimiter, job_queue: asyncio.Queue, num_retries=2
):
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
                break
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


async def consume(
    job: EmbedJob, rate_limiter: AsyncLimiter, job_queue: asyncio.Queue, num_retries=2
):
    """
    Consumer task that consumes batches from the queue and generates embeddings.
    """
    logging.debug("Starting consumer task")
    return await asyncio.gather(
        *[
            worker(job, rate_limiter, job_queue, num_retries)
            for _ in range(job.max_concurrent_requests)
        ],
        return_exceptions=True,
    )


async def run(job: EmbedJob):
    """
    Main entry point for the embedding job.
    """
    job_queue = asyncio.Queue(maxsize=job.max_concurrent_requests)
    request_limiter = AsyncLimiter(config.max_requests_per_minute(job.backend), 60)
    with Live(auto_refresh=True) as live:
        live.console.rule("Embedding Job: " + job.job_id)
        ui_task = asyncio.create_task(textui.render_ui(job, live))
        producer_task = asyncio.create_task(produce(job, job_queue))
        consumer_task = asyncio.create_task(
            consume(job, request_limiter, job_queue=job_queue)
        )
        try:
            await asyncio.wait({producer_task}, return_when=asyncio.FIRST_EXCEPTION)
            await job_queue.join()
            await terminate(consumer_task)
        except KeyboardInterrupt:
            await terminate(producer_task, consumer_task)
        finally:
            await client.cleanup()
            await ui_task
