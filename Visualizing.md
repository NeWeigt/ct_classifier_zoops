# Visualizing — Dataset visualization for ct_classifier_zoops

This document explains how the `Visualizing.py` utility works, and provides a
step-by-step annotated walkthrough plus the full script. Use this to explore
the dataset contents, save sample image grids, and inspect class distributions
quickly.

## Purpose

- Quickly sample images from the dataset and save a grid of thumbnails.
- Display class labels for each sample so you can get a feel for the data.

## Prerequisites

- Python 3.8+ preferred
- Install the minimal deps required by the script:

```bash
pip install PyYAML pillow matplotlib
```

Also ensure project's local requirements (e.g. `torch`, `torchvision`) are installed
if you want the script to import the `CTDataset` class in `ct_classifier.dataset`.

## Quick usage

Run the script from the `ct_classifier_zoops` folder (or provide full paths):

```bash
python Visualizing.py --config configs/exp_resnet18.yaml --n-samples 16 --out-dir viz_out
```

- `--config` : path to the YAML config (uses `data_root` inside the file).
- `--n-samples` : how many images to sample and include in the grid.
- `--out-dir` : where to save the resulting PNG (defaults to `viz_out`).
- `--show` : optional, if provided the script will open an interactive window.

Example saved file: `viz_out/samples_train.png`.

## Annotated step-by-step explanation

1) Load YAML config

- The script reads the YAML experiment config to get `data_root` and
  `image_size`. This keeps the visualization consistent with how the dataset
  is configured for training.

2) Try to import project's `CTDataset`

- The script prefers using `ct_classifier.dataset.CTDataset` so it samples the
  exact training split and follows any dataset indexing the project uses.
- If that import fails (missing torch, wrong path, etc.), the script falls
  back to reading the COCO-style JSON annotation files directly and builds a
  simple list of filename/label pairs.

3) Sample image entries

- If `CTDataset` is available, random indices are drawn and the dataset's
  internal `data` list is used to obtain filenames and labels.
- Otherwise the script parses the annotation JSON (e.g. `train_annotations.json`)
  and picks the first annotation per image (matching project's behavior).

4) Load image files

- Images are opened using Pillow from `data_root/eccv_18_all_images_sm/`.
- Missing images are reported and skipped.

5) Create PNG grid

- The selected images are arranged in a approximately square grid (max 6
  columns) and saved to `--out-dir` as `samples_<split>.png`.

## Full annotated script

Below is the full `Visualizing.py` content. You can copy it directly or open
the file at [Visualizing.py](Visualizing.py).

```python
#!/usr/bin/env python3
"""
Simple dataset visualization tool for the CT classifier project.

Usage examples:
  python Visualizing.py --config configs/exp_resnet18.yaml --n-samples 12 --out-dir viz_out

This script will sample images from the dataset, show them in a grid and
save a PNG with class labels. It uses the project's `ct_classifier` dataset
loader when possible and falls back to reading the COCO-style JSON.
"""
import argparse
import os
import random
import sys
from math import ceil

from PIL import Image
import matplotlib.pyplot as plt

try:
    import yaml
except Exception:
    print('PyYAML is required. Install with `pip install PyYAML`')
    raise


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def ensure_import_path():
    # Add the script's parent folder to sys.path so `ct_classifier` can be imported
    base = os.path.dirname(os.path.abspath(__file__))
    if base not in sys.path:
        sys.path.insert(0, base)


def load_category_names(cfg, split='train'):
    annoPath = os.path.join(cfg['data_root'], 'eccv_18_annotation_files',
                            'train_annotations.json' if split == 'train' else 'cis_val_annotations.json')
    import json
    meta = json.load(open(annoPath, 'r'))
    return [c.get('name', str(i)) for i, c in enumerate(meta['categories'])]


def main():
    parser = argparse.ArgumentParser(description='Visualize CT dataset samples')
    parser.add_argument('--config', '-c', required=True, help='Path to YAML config')
    parser.add_argument('--split', default='train', choices=['train', 'val'], help='Dataset split')
    parser.add_argument('--n-samples', type=int, default=16, help='Number of sample images')
    parser.add_argument('--out-dir', default='viz_out', help='Directory to save outputs')
    parser.add_argument('--show', action='store_true', help='Show interactive window')
    args = parser.parse_args()

    cfg = load_config(args.config)

    ensure_import_path()
    try:
        from ct_classifier.dataset import CTDataset
        dataset = CTDataset(cfg, split=args.split)
        use_dataset = True
    except Exception:
        dataset = None
        use_dataset = False

    os.makedirs(args.out_dir, exist_ok=True)

    # Prepare category names
    cat_names = load_category_names(cfg, split=args.split)

    # Choose sample indices
    if use_dataset:
        total = len(dataset)
        indices = random.sample(range(total), min(args.n_samples, total))
        entries = [(dataset.data[i][0], dataset.data[i][1]) for i in indices]
    else:
        # fallback: read annotations directly
        import json
        annoPath = os.path.join(cfg['data_root'], 'eccv_18_annotation_files',
                                'train_annotations.json' if args.split == 'train' else 'cis_val_annotations.json')
        meta = json.load(open(annoPath, 'r'))
        images = dict([[i['id'], i['file_name']] for i in meta['images']])
        labels_map = dict([[c['id'], idx] for idx, c in enumerate(meta['categories'])])
        # take one annotation per image
        images_covered = set()
        entries = []
        for anno in meta['annotations']:
            imgID = anno['image_id']
            if imgID in images_covered:
                continue
            fname = images[imgID]
            lab = labels_map[anno['category_id']]
            entries.append((fname, lab))
            images_covered.add(imgID)
        if len(entries) == 0:
            raise RuntimeError('No dataset entries found')
        random.shuffle(entries)
        entries = entries[:args.n_samples]

    # Load images and plot
    imgs = []
    for fname, lab in entries:
        img_path = os.path.join(cfg['data_root'], 'eccv_18_all_images_sm', fname)
        if not os.path.exists(img_path):
            print('Missing image:', img_path)
            continue
        im = Image.open(img_path).convert('RGB')
        imgs.append((im, lab, fname))

    if len(imgs) == 0:
        print('No images available to visualize.')
        return

    n = len(imgs)
    cols = min(6, int(ceil(n**0.5)))
    rows = int(ceil(n / cols))
    fig_w = cols * 3
    fig_h = rows * 3
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    axes = axes.flatten() if n > 1 else [axes]

    for ax in axes[n:]:
        ax.axis('off')

    for i, (im, lab, fname) in enumerate(imgs):
        ax = axes[i]
        ax.imshow(im)
        title = f"{lab}: {cat_names[lab] if lab < len(cat_names) else 'cls'+str(lab)}"
        ax.set_title(title, fontsize=8)
        ax.axis('off')

    out_path = os.path.join(args.out_dir, f'samples_{args.split}.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print('Saved visualization to', out_path)

    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
```

## Troubleshooting

- If the script prints `Missing image: ...`, verify `data_root` in your YAML
  and that the `eccv_18_all_images_sm` folder contains the listed files.
- If import of `ct_classifier.dataset` fails, make sure you run the script from
  the project root or the script's folder so local imports resolve.

## Next steps

- Run the script and inspect `viz_out/samples_train.png`.
- If you'd like, I can run a smoke test here and save a sample visualization.
