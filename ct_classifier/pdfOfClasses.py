#!/usr/bin/env python

"""Utilities to export class-specific image PDFs.

Two main modes:

1. Ground-truth mode (default):

     - Creates a PDF of **all images of a given ground-truth class**
         ("true_label").
     - Each image is annotated with the **model-predicted class**.

2. Predicted-class mode:

     - Creates a PDF of **all images of a given predicted class**
         ("pred_class_label").
     - Each image is annotated with the **ground-truth class** above it.

Example usage from the ct_classifier folder:

        # All images whose *true* label is "shrimp"
        python pdfOfClasses.py shrimp

        # All images whose *predicted* label is "shrimp"
        python pdfOfClasses.py shrimp --mode pred
"""

import argparse
import math
import os
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd


def _load_and_downsample_image(img_path: str, max_side: int = 256):
    """Load an image and downsample it so the longest side is <= max_side.

    This reduces the size of the embedded images in the PDF and helps avoid
    extremely large files or memory issues, while keeping enough detail to
    visually inspect predictions.
    """

    img = plt.imread(img_path)

    # If image is already small enough, return as-is
    if img.ndim < 2:
        return img

    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return img

    # Simple downsampling by striding; avoids extra dependencies.
    stride = max(1, int(math.ceil(longest / max_side)))
    return img[::stride, ::stride, ...]


def save_true_class_images_to_pdf(
    classified_predicted: pd.DataFrame,
    true_label: str,
    pdf_path: Optional[str] = None,
    n_cols: int = 5,
    rows_per_page: int = 8,
    cell_size: float = 1.0,
) -> str:

    """Save all images of a given *ground-truth* class to a (multi-page) PDF.

    Parameters
    ----------
    classified_predicted : pd.DataFrame
        DataFrame with at least 'image_path' and 'true_label' columns. If
        'pred_class_label' is present it will be overlaid as text.
    true_label : str
        The *ground-truth* class of organisms to export (e.g. "shrimp", "other").
    pdf_path : str, optional
        Output PDF path. If None, defaults to
        "true_{true_label}_images_grayscale.pdf".
    n_cols : int, optional
        Number of image columns per page.
    rows_per_page : int, optional
        Maximum number of rows of images per page.
    cell_size : float, optional
        Size of each image cell in inches.

    Returns
    -------
    str
        Absolute path to the created PDF.
    """

    if "true_label" not in classified_predicted.columns:
        raise ValueError("DataFrame must contain a 'true_label' column for this mode.")

    df = classified_predicted[classified_predicted["true_label"] == true_label]
    if df.empty:
        raise ValueError(f"No rows with true_label == '{true_label}'")

    n_images = len(df)
    images_per_page = n_cols * rows_per_page

    if pdf_path is None:
        pdf_path = os.path.join(
            "/home/Nele/code", f"true_{true_label}.pdf"
        )

    with PdfPages(pdf_path) as pdf:
        for start in range(0, n_images, images_per_page):
            end = min(start + images_per_page, n_images)
            batch_df = df.iloc[start:end]
            batch_size = len(batch_df)
            n_rows = math.ceil(batch_size / n_cols)

            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(n_cols * cell_size, n_rows * cell_size),
            )
            axes = axes.flatten()

            for ax, (_, row) in zip(axes, batch_df.iterrows()):
                img_path = row["image_path"]
                if not os.path.exists(img_path):
                    ax.axis("off")
                    continue

                img = _load_and_downsample_image(img_path)
                ax.imshow(img, rasterized=True)

                # Always show the ground-truth on top; optionally show prediction
                title_lines = [f"True: {row['true_label']}"]
                if "pred_class_label" in row.index:
                    title_lines.append(f"Pred: {row['pred_class_label']}")

                ax.set_title("\n".join(title_lines), fontsize=6)
                ax.axis("off")

            # turn off any unused axes on the last page
            for ax in axes[batch_size:]:
                ax.axis("off")

            plt.tight_layout()
            pdf.savefig(fig, dpi=100, bbox_inches="tight")
            plt.close(fig)

    abs_path = os.path.abspath(pdf_path)
    print(f"Saved compressed PDF: {abs_path}")
    return abs_path


def save_predicted_class_images_to_pdf(
    classified_predicted: pd.DataFrame,
    predicted_class: str,
    pdf_path: Optional[str] = None,
    n_cols: int = 5,
    rows_per_page: int = 8,
    cell_size: float = 1.0,
) -> str:

    """Save all images of a given *predicted* class to a (multi-page) PDF.

    Parameters
    ----------
    classified_predicted : pd.DataFrame
        DataFrame with at least 'image_path' and 'pred_class_label' columns. If
        'true_label' is present it will be shown above the prediction.
    predicted_class : str
        The *predicted* class of organisms to export (e.g. "shrimp", "other").
    pdf_path : str, optional
        Output PDF path. If None, defaults to
        "pred_{predicted_class}_images_grayscale.pdf".
    n_cols : int, optional
        Number of image columns per page.
    rows_per_page : int, optional
        Maximum number of rows of images per page.
    cell_size : float, optional
        Size of each image cell in inches.

    Returns
    -------
    str
        Absolute path to the created PDF.
    """

    if "pred_class_label" not in classified_predicted.columns:
        raise ValueError(
            "DataFrame must contain a 'pred_class_label' column for this mode."
        )

    df = classified_predicted[classified_predicted["pred_class_label"] == predicted_class]
    if df.empty:
        raise ValueError(f"No rows with pred_class_label == '{predicted_class}'")

    n_images = len(df)
    images_per_page = n_cols * rows_per_page

    if pdf_path is None:
        pdf_path = os.path.join(
            "/home/Nele/code", f"pred_{predicted_class}.pdf"
        )

    with PdfPages(pdf_path) as pdf:
        for start in range(0, n_images, images_per_page):
            end = min(start + images_per_page, n_images)
            batch_df = df.iloc[start:end]
            batch_size = len(batch_df)
            n_rows = math.ceil(batch_size / n_cols)

            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(n_cols * cell_size, n_rows * cell_size),
            )
            axes = axes.flatten()

            for ax, (_, row) in zip(axes, batch_df.iterrows()):
                img_path = row["image_path"]
                if not os.path.exists(img_path):
                    ax.axis("off")
                    continue

                img = _load_and_downsample_image(img_path)
                ax.imshow(img, rasterized=True)

                # Show true label (if available) above the predicted class
                title_lines = []
                if "true_label" in row.index:
                    title_lines.append(f"True: {row['true_label']}")
                title_lines.append(f"Pred: {row['pred_class_label']}")

                ax.set_title("\n".join(title_lines), fontsize=6)
                ax.axis("off")

            # turn off any unused axes on the last page
            for ax in axes[batch_size:]:
                ax.axis("off")

            plt.tight_layout()
            pdf.savefig(fig, dpi=100, bbox_inches="tight")
            plt.close(fig)

    abs_path = os.path.abspath(pdf_path)
    print(f"Saved compressed PDF: {abs_path}")
    return abs_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Create PDFs of class-specific images from a predictions CSV. "
            "By default, filters by ground-truth class (true_label) and "
            "overlays the model prediction. Use --mode pred to filter by "
            "predicted class and show the ground truth above it."
        )
    )
    parser.add_argument(
        "class_name",
        help=(
            "Class name to filter on. Interpreted as a ground-truth "
            "class when --mode true (default), or as a predicted class "
            "when --mode pred."
        ),
    )
    parser.add_argument(
        "--mode",
        help=(
            "Filtering mode: 'true' = filter by ground-truth true_label "
            "and overlay predictions; 'pred' = filter by predicted "
            "pred_class_label and show the ground truth above it."
        ),
        dest="mode",
        choices=["true", "pred"],
        default="true",
    )
    parser.add_argument(
        "--csv-path",
        default="/mnt/class_data/Nele/ROIs_August2021/predictions_resnet50_St02.csv",
        help="Path to the CSV with predictions (default is the August 2021 file)",
    )

    args = parser.parse_args()

    csv_path = args.csv_path
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        class_name = args.class_name

        if args.mode == "true":
            # Ensure we have a true_label column; if not, try to infer it
            # from the image_path (as in the original script).
            if "true_label" not in df.columns:
                if "image_path" in df.columns:
                    df["true_label"] = df["image_path"].apply(
                        lambda x: x.split("/")[-1].split("__")[0]
                    )
                else:
                    raise ValueError(
                        "CSV must contain a 'true_label' column or an 'image_path' "
                        "column to infer true_label for ground-truth mode."
                    )

            save_true_class_images_to_pdf(df, true_label=class_name)

        else:  # args.mode == "pred"
            if "pred_class_label" not in df.columns:
                raise ValueError(
                    "CSV must contain a 'pred_class_label' column with predicted "
                    "classes for predicted-class mode."
                )

            # If true_label is missing but we have image_path, infer it so
            # the predicted-class PDF can still display the ground truth.
            if "true_label" not in df.columns and "image_path" in df.columns:
                df["true_label"] = df["image_path"].apply(
                    lambda x: x.split("/")[-1].split("__")[0]
                )

            save_predicted_class_images_to_pdf(df, predicted_class=class_name)
    else:
        print(f"CSV not found: {csv_path}")

