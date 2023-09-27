"""
Configuration parameters for clients
"""
import functools

from etypes import Backend

# Scale down factor
SCALE_DOWN_FACTOR = 0.8

max_requests_limits = {
    Backend.OPENAI: 10000,
    Backend.COHERE: 1500, # Using trial key values
    Backend.HUGGINGFACE: 100
}

RATE_LIMIT_WAIT_TIME_SECS = 0.5

max_token_limits = {
    Backend.OPENAI: 8191,
    Backend.COHERE: 8000,
    Backend.HUGGINGFACE: 512
}

# TODO: Figure out what the max batch size is for each service
# Code: https://github.com/openai/openai-python/blob/main/openai/embeddings_utils.py#L43
MAX_BATCH_SIZE = 16


@functools.cache
def max_requests_per_minute(backend: Backend) -> int:
    base_rpm = max_requests_limits.get(backend, 100)
    return int(base_rpm * SCALE_DOWN_FACTOR)

def batch_size(backend: Backend) -> int:
    # TODO: backend aware batch size
    return MAX_BATCH_SIZE

def max_tokens(backend: Backend) -> int:
    return max_token_limits.get(backend, 512)
