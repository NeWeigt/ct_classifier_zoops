'''
    PyTorch dataset class for COCO-CT-formatted datasets. Note that you could
    use the official PyTorch MS-COCO wrappers:
    https://pytorch.org/vision/master/generated/torchvision.datasets.CocoDetection.html

    We just hack our way through the COCO JSON files here for demonstration
    purposes.

    See also the MS-COCO format on the official Web page:
    https://cocodataset.org/#format-data

    2022 Benjamin Kellenberger
'''

import os
import json
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image


class CTDataset(Dataset):

    def __init__(self, cfg, split='train'):
        '''
            Constructor. Here, we collect and index the dataset inputs and
            labels.
        '''
        self.data_root = cfg['data_root']
        self.annoPath = cfg['annotation_file']
        self.split = split
        # Transforms. Here's where we could add data augmentation 
        #  For now, we just resize the images to the same dimensions...and convert them to torch.Tensor.
        #  For other transformations see Björn's lecture on August 11 or 
        self.transform = Compose([              
            Resize((cfg['image_size'])),        
            ToTensor()                          
        ])
        
        # index data into list
        self.data = []

        # load annotation file
        # annoPath = os.path.join(
        #     self.data_root,
        #     'eccv_18_annotation_files',
        #     'train_annotations.json' if self.split=='train' else 'cis_val_annotations.json'
        # )        
        print(f"Loading annotations from {self.annoPath}..."
              )
        print(f"{os.path.exists(self.annoPath)=}")
        annoPath = self.annoPath
        meta = json.load(open(annoPath, 'r'))
        meta_split = meta[split]
        images = meta_split['images']
        annotations = meta_split['annotations']

        # enable filename lookup. Creates image IDs and assigns each ID one filename. 
        #  If your original images have multiple detections per image, this code assumes
        #  that you've saved each detection as one image that is cropped to the size of the
        #  detection, e.g., via megadetector.
        # images = dict([[i['id'], i['image_name']] for i in meta['images']])
        # # create custom indices for each category that start at zero. Note: if you have already
        # #  had indices for each category, they might not match the new indices.
        # labels = dict([[c['id'], idx] for idx, c in enumerate(meta['categories'])])

        stat = list(set([im.split('/')[0] for im in images]))
        stat.sort()
        print(f"{stat=}")
        print(f"{images[:5]=}")

        print(f"{len(images)=}")
        print(f"{annotations[:5]=}")
        print(f"{len(annotations)=}")

        # load label mapping from category ID to index starting at 0
        labels_mapping_path = "/home/Nele/code/scripts/DataPrep_Classifier/category_dict.json"
        with open(labels_mapping_path, 'r') as f:
            label_mapping = json.load(f)

        labels = [label_mapping[anno] for anno in annotations]

        print(f"{annotations[:5]=}")
        print(f"{labels[:5]=}")

        self.data = list(zip(images, labels))
    
        
        # since we're doing classification, we're just taking the first annotation per image and drop the rest
        # images_covered = set()      # all those images for which we have already assigned a label
        # for anno in meta['annotations']:
        #     imgID = anno['image_id']
        #     if imgID in images_covered:
        #         continue
            
        #     # append image-label tuple to data
        #     imgFileName = images[imgID]
        #     label = anno['category_id']
        #     labelIndex = labels[label]
        #     self.data.append([imgFileName, labelIndex])
        #     images_covered.add(imgID)       # make sure image is only added once to dataset
    

    def __len__(self):
        '''
            Returns the length of the dataset.
        '''
        return len(self.data)

    
    def __getitem__(self, idx):
        '''
            Returns a single data point at given idx.
            Here's where we actually load the image.
        '''
        image_name, label = self.data[idx]              # see line 57 above where we added these two items to the self.data list

        # load image
        image_path = os.path.join(self.data_root, image_name)
        img = Image.open(image_path).convert('RGB')     # the ".convert" makes sure we always get three bands in Red, Green, Blue order

        # transform: see lines 31ff above where we define our transformations
        img_tensor = self.transform(img)

        return img_tensor, label, image_name  # return image name for debugging purposes only