"""Select and save the best epoch checkpoint based on validation accuracy.

This script assumes that training checkpoints are stored in the
`model_states/` directory and that each checkpoint is a PyTorch
`torch.save` dictionary containing at least the key `oa_val` (validation
overall accuracy) and `model` (state_dict).

Usage (from project root):

    python -m ct_classifier.save_best_epoch

or directly:

    python ct_classifier/save_best_epoch.py
"""

import os
import glob
import shutil
from typing import Optional, Tuple

import torch


def find_best_checkpoint(directory: str = "model_states") -> Optional[Tuple[str, float]]:
    """Return path and best oa_val among all .pt files in `directory`.

    If no checkpoints are found or none contain `oa_val`, returns None.
    """
    pattern = os.path.join(directory, "*.pt")
    checkpoint_paths = sorted(glob.glob(pattern))

    if not checkpoint_paths:
        print(f"No checkpoints found in '{directory}'.")
        return None

    best_path: Optional[str] = None
    best_oa: float = float("-inf")

    for path in checkpoint_paths:
        try:
            checkpoint = torch.load(path, map_location="cpu")
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Skipping '{path}' (failed to load): {exc}")
            continue

        oa_val = checkpoint.get("oa_val")
        if oa_val is None:
            print(f"Skipping '{path}' (no 'oa_val' key).")
            continue

        if oa_val > best_oa:
            best_oa = oa_val
            best_path = path

    if best_path is None:
        print("No checkpoint with 'oa_val' found.")
        return None

    return best_path, best_oa


def save_best_checkpoint(directory: str = "model_states", output_name: str = "best.pt") -> None:
    """Find the best checkpoint and save/copy it as `output_name` in `directory`."""
    result = find_best_checkpoint(directory)
    if result is None:
        return

    best_path, best_oa = result
    output_path = os.path.join(directory, output_name)

    # Simply copy the file so we keep the exact contents
    shutil.copyfile(best_path, output_path)

    print(f"Best checkpoint: '{best_path}' with oa_val={best_oa:.4f}")
    print(f"Saved as '{output_path}'.")


if __name__ == "__main__":
    save_best_checkpoint()
