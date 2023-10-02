import json
from pathlib import Path


def line(fname: Path):
    with fname.open() as f:
        for line in f:
            yield line.strip()

def jsonl(fname: Path):
    for nxt_line in line(fname):
        yield json.loads(nxt_line)
