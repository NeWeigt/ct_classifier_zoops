"""Select and save the best epoch checkpoint based on validation accuracy.

This script assumes that training checkpoints are stored in the
`model_states/` directory and that each checkpoint is a PyTorch
`torch.save` dictionary containing at least the key `oa_val` (validation
overall accuracy) and `model` (state_dict).

Usage (from project root):

    python -m ct_classifier.save_best_epoch \
        --input-dir model_states \
        --output-dir model_states \
        --output-name best.pt

or directly:

    python ct_classifier/save_best_epoch.py --input-dir path/to/checkpoints
"""
## Best checkpoint: '/home/Nele/code/ct_classifier_zoops/model_states/159.pt' with oa_val=0.8395
# Saved as '/home/Nele/code/ct_classifier_zoops/best_epochs/first_training_resnet18.pt'.

import argparse
import os
import glob
import shutil
from typing import Optional, Tuple

import torch


def find_best_checkpoint(directory = '/home/Nele/code/ct_classifier_zoops/model_states') -> Optional[Tuple[str, float]]:
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


def save_best_checkpoint(
    input_dir = '/home/Nele/code/ct_classifier_zoops/model_states',
    output_dir = '/home/Nele/code/ct_classifier_zoops/best_epochs',
    output_name: str = "first_training_resnet18.pt",
) -> None:
    """Find the best checkpoint and save/copy it to the desired location.

    Parameters
    ----------
    input_dir: str
        Directory to search for checkpoint ``.pt`` files.
    output_dir: Optional[str]
        Directory to save the best checkpoint. If ``None``, defaults to
        ``input_dir``.
    output_name: str
        File name for the copied best checkpoint.
    """

    result = find_best_checkpoint(input_dir)
    if result is None:
        return

    best_path, best_oa = result
    if output_dir is None:
        output_dir = input_dir

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_name)

    # Simply copy the file so we keep the exact contents
    shutil.copyfile(best_path, output_path)

    print(f"Best checkpoint: '{best_path}' with oa_val={best_oa:.4f}")
    print(f"Saved as '{output_path}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Find the best checkpoint by validation accuracy and copy it "
            "to a specified directory."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="model_states",
        help="Directory containing checkpoint .pt files (default: model_states)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory to save the best checkpoint (default: same as --input-dir)"
        ),
    )
    parser.add_argument(
        "--output-name",
        default="best.pt",
        help="File name for the best checkpoint (default: best.pt)",
    )

    args = parser.parse_args()

    save_best_checkpoint(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        output_name=args.output_name,
    )
