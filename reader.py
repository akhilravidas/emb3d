import json
from pathlib import Path


def jsonl(fname: Path):
    with fname.open() as f:
        for line in f:
            yield json.loads(line.strip())
