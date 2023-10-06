"""
Readers
"""
import json
from typing import Iterator, TextIO


def line(f_io: TextIO) -> Iterator[str]:
    for line in f_io:
        yield line.strip()


def jsonl(f_io: TextIO) -> Iterator[dict]:
    for nxt_line in line(f_io):
        yield json.loads(nxt_line)
