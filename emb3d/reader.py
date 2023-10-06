"""
Readers
"""
import json
from pathlib import Path
from typing import Iterator, TextIO

# def line(fname: Path):
#     with fname.open() as f:
#         for line in f:
#             yield line.strip()


# def jsonl(fname: Path):
#     for nxt_line in line(fname):
#         yield json.loads(nxt_line)


def line(f_io: TextIO) -> Iterator[str]:
    for line in f_io:
        yield line.strip()


def jsonl(f_io: TextIO) -> Iterator[dict]:
    for nxt_line in line(f_io):
        yield json.loads(nxt_line)
