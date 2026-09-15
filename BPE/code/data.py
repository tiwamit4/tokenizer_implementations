"""Batch-read local Parquet, CSV, JSONL, or plain-text training data."""

import csv
import json
from pathlib import Path

from .config import DEFAULT_BATCH_SIZE, DEFAULT_TEXT_COLUMN


def iter_texts(paths, text_column=DEFAULT_TEXT_COLUMN, batch_size=DEFAULT_BATCH_SIZE, max_texts=None):
    """Yield nonempty texts without loading complete files into memory."""
    if batch_size <= 0 or (max_texts is not None and max_texts <= 0):
        raise ValueError("Batch size and text limit must be positive")
    emitted = 0
    for path in paths:
        for text in _read_file(Path(path), text_column, batch_size):
            if not isinstance(text, str) or not text.strip():
                continue
            yield text
            emitted += 1
            if max_texts is not None and emitted >= max_texts:
                return


def _read_file(path, column, batch_size):
    if path.suffix.lower() == ".parquet":
        try:
            import pyarrow.parquet as parquet
        except ImportError as error:
            raise ImportError("Parquet input requires pyarrow; install BPE/requirements.txt") from error
        for batch in parquet.ParquetFile(path).iter_batches(
            batch_size=batch_size, columns=[column]
        ):
            yield from batch.column(0).to_pylist()
    elif path.suffix.lower() in {".csv", ".jsonl", ".txt"}:
        with path.open(encoding="utf-8", newline="") as file:
            if path.suffix.lower() == ".csv":
                for row in csv.DictReader(file):
                    yield row[column]
            elif path.suffix.lower() == ".jsonl":
                for line in file:
                    if line.strip():
                        yield json.loads(line)[column]
            else:
                yield from file
    else:
        raise ValueError(f"Unsupported input format: {path}")
