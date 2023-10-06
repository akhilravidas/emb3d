import json
import logging
from typing import Iterator

from emb3d import client, reader
from emb3d.types import Batch, EmbedJob


def write_batch_results_post_lock(job: EmbedJob, batch: Batch):
    """
    Write the results of a batch to the output file, assumes calling context has
    ensured that there is atmost one writer writing to the output file.
    """
    logging.debug("Writing computed batch results, size = [%d]", len(batch.row_ids))
    for idx, _ in enumerate(batch.row_ids):
        job.out_file.write(
            json.dumps(
                {
                    "row_id": batch.row_ids[idx],
                    "input": batch.inputs[idx],
                    "embedding": batch.embeddings[idx]
                    if batch.embeddings is not None
                    else None,
                    "error": str(batch.error) if batch.error else None,
                }
            )
            + "\n"
        )
    job.batch_saved(len(batch.row_ids))


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

        can_merge_token = batch_token_count + new_tokens < max_tokens
        can_merge_element = len(batch_ids) + 1 < batch_size

        can_merge = len(batch_ids) == 0 or (can_merge_token and can_merge_element)
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
