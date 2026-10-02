"""Read local Parquet text in batches."""

from pathlib import Path


def iter_texts(directory, pattern, column, batch_size, max_texts):
    paths = sorted(Path(directory).glob(pattern))
    if not paths:
        raise FileNotFoundError("No local Parquet training files found")
    if batch_size <= 0 or (max_texts is not None and max_texts <= 0):
        raise ValueError("Batch size and row limit must be positive")
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise ImportError("Install SentencePiece/requirements.txt") from error
    emitted = 0
    for path in paths:
        for batch in parquet.ParquetFile(path).iter_batches(
                batch_size=batch_size, columns=[column]):
            for text in batch.column(0).to_pylist():
                if not isinstance(text, str) or not text.strip():
                    continue
                yield text
                emitted += 1
                if max_texts is not None and emitted >= max_texts:
                    return
