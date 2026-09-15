"""Train BPE on local data using the settings in config.py."""

from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from BPE.code import config
from BPE.code.data import iter_texts
from BPE.code.model_io import save_model
from BPE.code.optimized_training import train_bpe_optimized


def main():
    dataset = Path(config.DEFAULT_TRAINING_DIR)
    output = Path(config.DEFAULT_MODEL_PATH)
    if output.exists():
        raise FileExistsError(
            "Output already exists; change DEFAULT_MODEL_PATH in config.py"
        )
    files = (
        [dataset] if dataset.is_file()
        else sorted(dataset.glob(config.DEFAULT_TRAINING_PATTERN))
    )
    if not files:
        raise FileNotFoundError(
            "No training files found; check DEFAULT_TRAINING_DIR and "
            "DEFAULT_TRAINING_PATTERN in config.py"
        )
    texts = iter_texts(
        files,
        text_column=config.DEFAULT_TEXT_COLUMN,
        batch_size=config.DEFAULT_BATCH_SIZE,
        max_texts=None if config.USE_ALL_TEXTS else config.DEFAULT_MAX_TEXTS,
    )
    print(f"Training on {len(files)} file(s); merges={config.DEFAULT_NUM_MERGES}")
    merges, tokens = train_bpe_optimized(texts, config.DEFAULT_NUM_MERGES)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_model(output, merges, tokens, config.DEFAULT_NUM_MERGES)
    print(f"Learned {len(merges)} merges and {len(tokens)} tokens")
    print(f"Saved: {output}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, ImportError, KeyError) as error:
        sys.exit(f"Training failed: {error}")
