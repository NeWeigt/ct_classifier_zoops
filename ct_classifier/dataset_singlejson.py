'''
    PyTorch dataset class for simple split JSON (singleImage.json) where the
    annotation file contains keys like 'train', 'val', 'test' and each key maps
    to an object with parallel 'images' and optional 'annotations' lists.

    The style and API mirror `dataset.py` to make swapping loaders straightforward.

    2026 Adapted for singleImage.json
'''

import os
import json
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image


class CTDatasetSingleJSON(Dataset):

    def __init__(self, cfg, split='train'):
        '''
            Constructor. Collects and indexes dataset inputs and labels from
            a compact split-style JSON file.
        '''
        self.data_root = cfg['data_root']
        self.image_root = cfg['image_root']
        self.split = split

        # Transforms: keep same style as original dataset.py
        self.transform = Compose([
            Resize((cfg['image_size'])),
            ToTensor()
        ])

        # index data into list
        self.data = []

        # load annotation file (allow override via cfg)
        annoPath = cfg.get('annotation_file') or os.path.join(self.data_root,  'singleImage.json')
        meta = json.load(open(annoPath, 'r'))

                # Expecting structure: { 'train': { 'images': [...], 'annotations': [...] }, ... }
        if not isinstance(meta, dict) or self.split not in meta:
            raise ValueError(f"annotation file does not contain split '{self.split}': {annoPath}")

        split_obj = meta[self.split]
        images = split_obj.get('images', [])
        annotations = split_obj.get('annotations', None)

        # If annotations list is provided and matches images length, use it directly
        if annotations and len(annotations) == len(images):
            # map category string -> index
            labels_map = {}
            for cat in annotations:
                if cat not in labels_map:
                    labels_map[cat] = len(labels_map)#give the new category the last index
            for img_name, cat in zip(images, annotations):
                self.data.append([img_name, labels_map[cat]]) # creates pairs of image_name and catogory_id(number) 
        else:
            raise ValueError(f"annotations missing or length mismatch in split '{self.split}' of {annoPath}")
            # infer category from filename prefix before '__'
            # labels_map = {}
            # for img_name in images:
            #     if '__' in img_name:
            #         cat = img_name.split('__')[0]
            #     else:
            #         cat = os.path.splitext(img_name)[0]
            #     if cat not in labels_map:
            #         labels_map[cat] = len(labels_map)
            #     self.data.append([img_name, labels_map[cat]])


    def __len__(self):
        '''
            Returns the length of the dataset.
        '''
        return len(self.data)


    def __getitem__(self, idx):
        '''
            Returns a single data point at given idx. Loads the image file and
            applies the same transforms used in `dataset.py`.
        '''
        image_name, label = self.data[idx]

        # resolve image path (same location as original dataset.py)
        image_path = os.path.join(self.image_root, image_name)

        img = Image.open(image_path).convert('RGB')

        # transform
        img_tensor = self.transform(img)

        return img_tensor, label
