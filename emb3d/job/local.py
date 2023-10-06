import sentence_transformers

from emb3d import config
from emb3d.job.common import gen_batch, write_batch_results_post_lock
from emb3d.types import EmbedJob


def run(job: EmbedJob):
    """
    Run the job.
    """
    model = sentence_transformers.SentenceTransformer(job.model_id)
    for batch in gen_batch(job, job.batch_size, config.max_tokens(job.backend)):
        batch.embeddings = model.encode(batch.inputs).tolist()
        batch.error = None
        write_batch_results_post_lock(job, batch)
