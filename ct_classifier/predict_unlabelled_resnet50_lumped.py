"""Run the best ResNet-50 checkpoint on a folder of unlabelled images.

This script is intentionally simple and case-specific for this project.
It loads the checkpoint saved as ``best_epochs/first_training_resnet50.pt``
(which was selected via ``ct_classifier/save_best_epoch.py``) and applies
it to all images in a directory.

Optionally, you can "lump" (group) the original fine-grained classes
into coarser categories via a small JSON file that maps either original
label names (e.g. "salp", "copepod") or class indices (e.g. "5") to
group names (e.g. "crustaceans"). See the ``--lump-map`` argument.

Usage (from project root ``ct_classifier_zoops``):

    python ct_classifier/predict_unlabelled_resnet50_lumped.py \
        --image-dir /mnt/class_data/Nele/ROIs_August2024/Station01 \
        --output-csv /mnt/class_data/Nele/ROIs_August2024/predictions_resnet50_St01.csv \
        --lump-map /home/Nele/code/scripts/DataPrep_Classifier/class_lumping.json

"""

# Standard library imports for argument parsing, file handling and typing.

import argparse
import csv
import json
import os
from typing import Dict, Iterable, List, Optional, Tuple

# Progress bar for visual feedback during inference.
from tqdm import tqdm

# PyTorch and torchvision imports for model inference and image transforms.
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor

# Import the custom ResNet-50 architecture used during training.
from bigmodel import CustomResNet50


def get_repo_root() -> str:
    """Return absolute path to the project root (ct_classifier_zoops).

    We start from this file's directory and go one level up.
    """

    # Folder where this script lives: ct_classifier/
    here = os.path.dirname(os.path.abspath(__file__))
    # Go one directory up to reach the project root.
    return os.path.abspath(os.path.join(here, ".."))


def load_model(
    device: torch.device,
    num_classes: int = 28,
    checkpoint_name: str = "first_training_resnet50.pt",
) -> CustomResNet50:
    """Load CustomResNet50 with weights from the best-epoch checkpoint.

    This recreates the network architecture and fills it with trained weights.
    """

    # Build the full path to the checkpoint inside best_epochs/.
    repo_root = get_repo_root()
    checkpoint_path = os.path.join(repo_root, "best_epochs", checkpoint_name)

    # Fail fast with a clear error if the file is missing.
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at '{checkpoint_path}'. "
            "Make sure best_epochs/first_training_resnet50.pt exists."
        )

    # Initialize the ResNet-50 model with the correct number of classes.
    model = CustomResNet50(num_classes=num_classes)

    # Load the checkpoint dictionary from disk onto the chosen device.
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Some checkpoints store the state under the key "model"; otherwise
    # assume the checkpoint itself is a state_dict.
    state_dict = checkpoint.get("model", checkpoint)
    # Load the weights into the model architecture.
    model.load_state_dict(state_dict)

    # Move model to device (CPU or GPU) and set to evaluation mode.
    model.to(device)
    model.eval()
    return model


def load_label_mappings(
    category_dict_path: Optional[str] = None,
) -> Tuple[Optional[Dict[str, int]], Optional[Dict[int, str]]]:
    """Load label mapping used during training and build inverse mapping.

    The default path follows the project-specific setup.
    """

    if category_dict_path is None:
        # Default to the same mapping file that was used during training.
        category_dict_path = "/home/Nele/code/scripts/DataPrep_Classifier/category_dict.json"

    # If the mapping file is missing, fall back to numeric class indices only.
    if not os.path.exists(category_dict_path):
        print(
            f"WARNING: category_dict.json not found at '{category_dict_path}'. "
            "Predictions will only contain numeric class indices."
        )
        return None, None

    # Load mapping from human-readable label (string) to class index (int).
    with open(category_dict_path, "r") as f:
        label_to_idx: Dict[str, int] = json.load(f)

    # Build inverse mapping: from index back to label for nicer output.
    idx_to_label: Dict[int, str] = {v: k for k, v in label_to_idx.items()}
    return label_to_idx, idx_to_label


def load_lump_mapping(lump_map_path: Optional[str]) -> Optional[Dict[str, str]]:
    """Load an optional mapping that groups (lumps) classes together.

    The JSON file should be a simple dict where keys are either original
    class labels (e.g. "salp") or class indices as strings (e.g. "17"),
    and values are group names (e.g. "salps", "crustaceans").

    Example ``class_lumping.json``::

        {
            "salp": "salps",
            "salpchain": "salps",
            "17": "salps",  # by index
            "copepod": "crustaceans"
        }
    """

    if lump_map_path is None:
        return None

    if not os.path.exists(lump_map_path):
        print(f"WARNING: lump map file not found: {lump_map_path}; ignoring.")
        return None

    with open(lump_map_path, "r") as f:
        raw_map = json.load(f)

    # Normalize keys to strings so we can match both labels and indices easily.
    lump_map: Dict[str, str] = {str(k): str(v) for k, v in raw_map.items()}
    return lump_map


def iter_image_paths(root: str) -> Iterable[str]:
    """Yield image file paths under ``root`` (recursively).

    Only common image extensions are considered.
    """

    # We only consider files with these image extensions.
    exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")

    # Walk the directory tree and yield full paths to all matching images.
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in sorted(filenames):
            if name.lower().endswith(exts):
                yield os.path.join(dirpath, name)


def build_transform(image_size: Tuple[int, int] = (224, 224)):
    """Return the image transform used for inference.

    Mirrors the training setup: resize and convert to tensor.
    """
    # Resize images to the desired input size and convert to a tensor.
    return Compose([
        Resize(image_size),
        ToTensor(),
    ])


def predict_on_folder(
    image_dir: str,
    output_csv: str,
    device: torch.device,
    category_dict_path: Optional[str] = None,
    lump_map: Optional[Dict[str, str]] = None,
) -> None:
    """Run model on all images in ``image_dir`` and write predictions to CSV.

    This function ties everything together: it loads the model, loops over
    images, and stores one prediction per image in a CSV file.
    """

    # Basic sanity check: make sure we were given a directory.
    if not os.path.isdir(image_dir):
        raise NotADirectoryError(f"'{image_dir}' is not a directory.")

    print(f"Using device: {device}")

    # Load the trained model and the label mappings (if available).
    model = load_model(device=device)
    _label_to_idx, idx_to_label = load_label_mappings(category_dict_path)

    # Image transform matching the training preprocessing.
    transform = build_transform()

    # Collect all image paths we will process.
    image_paths = list(iter_image_paths(image_dir))
    if not image_paths:
        print(f"No images found in '{image_dir}'. Nothing to do.")
        return

    print(f"Found {len(image_paths)} images. Running inference...")

    # Create output folder if it does not exist yet.
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)) or ".", exist_ok=True)

    # Open the CSV file and write a header row.
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        header = [
            "image_path",
            "pred_class_index",
            "pred_class_label",
            "confidence",
        ]

        # If we have a lumping map, add an extra column for the group label.
        if lump_map is not None:
            header.append("lumped_label")

        writer.writerow(header)

        # Disable gradient computation to speed up inference and save memory.
        # Wrap the loop in a tqdm progress bar for visual confirmation.
        with torch.no_grad():
            for img_path in tqdm(image_paths, desc="Predicting", unit="img"):
                try:
                    # Open the image file and convert to RGB.
                    img = Image.open(img_path).convert("RGB")
                except Exception as exc:  # pragma: no cover - defensive
                    # If the file cannot be opened as an image, skip it.
                    print(f"Skipping '{img_path}' (failed to open): {exc}")
                    continue

                # Apply transforms and add a batch dimension.
                tensor = transform(img).unsqueeze(0).to(device)
                # Forward pass through the network: raw class scores.
                logits = model(tensor)
                # Convert scores to probabilities.
                probs = F.softmax(logits, dim=1)
                # Take the most confident class and its probability.
                conf, pred_idx = torch.max(probs, dim=1)

                # Convert 1-element tensors to plain Python types.
                idx = int(pred_idx.item())
                conf_val = float(conf.item())
                # Look up the human-readable label if we have a mapping.
                label = idx_to_label.get(idx) if idx_to_label is not None else ""

                row = [img_path, idx, label, conf_val]

                # Optionally compute a lumped/grouped label.
                if lump_map is not None:
                    lumped_label = ""
                    # Try original label first (if available), then index as string.
                    key_candidates = []
                    if label:
                        key_candidates.append(str(label))
                    key_candidates.append(str(idx))

                    for key in key_candidates:
                        if key in lump_map:
                            lumped_label = lump_map[key]
                            break

                    # If nothing matched, fall back to original label or index.
                    if not lumped_label:
                        lumped_label = label or str(idx)

                    row.append(lumped_label)

                # Write one CSV row per image.
                writer.writerow(row)

    print(f"Done. Predictions written to '{output_csv}'.")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for this script.

    This keeps ``main`` clean and documents how to call the script.
    """

    # Create an argument parser with a short description.
    parser = argparse.ArgumentParser(
        description="Run best ResNet-50 checkpoint on a folder of unlabelled images.",
    )
    # Folder containing the images to classify.
    parser.add_argument(
        "--image-dir",
        required=True,
        help="Path to folder containing unlabelled images (searched recursively).",
    )
    # Where to store the resulting CSV file with predictions.
    parser.add_argument(
        "--output-csv",
        default="predictions_resnet50_unlabelled.csv",
        help=(
            "Path to CSV file where predictions will be written "
            "(default: predictions_resnet50_unlabelled.csv)."
        ),
    )
    # Optional override for the category mapping JSON file.
    parser.add_argument(
        "--category-dict",
        default=None,
        help=(
            "Optional path to category_dict.json used during training. "
            "If omitted, uses the project-specific default."
        ),
    )
    # Optional mapping to lump (group) original classes into coarser groups.
    # The JSON file should map original labels or indices (as strings) to
    # group names, e.g. {"salp": "salps", "5": "crustaceans"}.
    parser.add_argument(
        "--lump-map",
        default=None,
        help=(
            "Optional JSON file mapping original classes to grouped classes "
            "(e.g. salp -> salps). If omitted, no lumping is applied."
        ),
    )
    # Optional override for the compute device.
    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Computation device, e.g. 'cuda' or 'cpu'. "
            "If omitted, uses 'cuda' when available, otherwise 'cpu'."
        ),
    )
    # Parse and return the populated Namespace.
    return parser.parse_args()


def main() -> None:
    """Entry point when the script is run from the command line."""

    # Read user-specified command line options.
    args = parse_args()

    # If the user did not specify a device, prefer CUDA when available.
    if args.device is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = args.device

    # Build a torch.device object from the string.
    device = torch.device(device_str)

    # Load optional lumping map if provided.
    lump_map = load_lump_mapping(args.lump_map)

    # Run prediction on the requested folder.
    predict_on_folder(
        image_dir=args.image_dir,
        output_csv=args.output_csv,
        device=device,
        category_dict_path=args.category_dict,
        lump_map=lump_map,
    )


if __name__ == "__main__":
    main()
