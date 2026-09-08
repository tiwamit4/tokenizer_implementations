"""Download WikiText-103 Raw into BPE/dataset.

Running this file performs the download; importing it does not.
"""

from pathlib import Path
import sys


# Make the BPE package importable when this script is run directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from BPE.code.config import DATASET_CONFIG_NAME, DATASET_REPO_ID


DATASET_DIR = Path(__file__).resolve().parent / "dataset"


def download_dataset() -> Path:
    """Download the configured WikiText subset and return its directory."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise SystemExit(
            f"huggingface_hub is not installed for: {sys.executable}\n"
            f"Install it with: \"{sys.executable}\" -m pip install "
            "huggingface_hub"
        ) from error

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {DATASET_REPO_ID}/{DATASET_CONFIG_NAME}")
    print(f"Destination: {DATASET_DIR}")

    snapshot_download(
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
        local_dir=DATASET_DIR,
        allow_patterns=[f"{DATASET_CONFIG_NAME}/*"],
    )

    print("Download completed.")
    return DATASET_DIR


if __name__ == "__main__":
    download_dataset()
