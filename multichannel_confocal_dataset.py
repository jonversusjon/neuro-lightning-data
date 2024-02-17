import os
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from PIL import Image, ImageSequence
from tifffile import TiffFile
from io import BytesIO
import json
import numpy
import random

import torch
from torchvision import transforms as T

import CheckSumManager


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, stage=None, transforms=None, inverse_transforms=None):
        self.img_dir = img_dir
        self.stage = stage  # 'fit', 'val', 'test'
        self.transforms = transforms
        self.inverse_transforms = inverse_transforms

        self.img_names = sorted([f for f in os.listdir(img_dir) if f.split('.')[-1] in ["jpg", "jpeg", "png", "tif", "tiff"]])

        self.checksummer = ChecksumManager()
        self.raw_hashes = [""] * len(self.img_names)
        self.transformed_hashes = [""] * len(self.img_names)

        self.fluo_channels = None
        first_img_path = os.path.join(self.img_dir, self.img_names[0])
        with Image.open(first_img_path) as img:
            # Loop through the pages/views in the TIFF file
            self.fluo_channels = 0
            for page in ImageSequence.Iterator(img):
                self.fluo_channels += 1

        self.indices = list(range(len(self)))
        self.device = device
        print(f"Initialized dataset with {len(self.img_names)} {self.fluo_channels}-channel images from {self.img_dir}")

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):

        filename = self.img_names[idx]
        img_path = os.path.join(self.img_dir, filename)

        raw_hash = self.raw_hashes[idx]
        transformed_hash = self.transformed_hashes[idx]

        with TiffFile(img_path) as tif:
            try:
                # Attempt to fetch and parse the 'ImageDescription' tag.
                metadata_str = tif.pages[0].tags['ImageDescription'].value
                metadata_dict = json.loads(metadata_str)

                # Extract information if available.
                treatment_code = metadata_dict.get('treatment_code', 0)
                treatment = metadata_dict.get('treatment', 'n/a')
            except (KeyError, json.JSONDecodeError):
                # Handle cases where 'ImageDescription' is missing or not valid JSON.
                treatment_code = 0
                treatment = 'n/a'

        with Image.open(img_path) as img:
            num_pages = img.n_frames
            # assert num_pages > 1, f"The image {filename} is not a multi-page TIFF. It has {num_pages} pages."

            images = []
            for i in range(num_pages):
                img.seek(i)
                page = img.copy()
                images.append(page)

            tensorize = Tensorize()
            tensor_images = [tensorize(page) for page in images]
            images = torch.stack(tensor_images, dim=0).squeeze(1)

            if raw_hash == "":
                buffer = BytesIO()
                torch.save(images, buffer)
                data = buffer.getvalue()
                raw_hash = self.checksummer.compute_hash(data)
                self.raw_hashes[idx] = raw_hash

            images = self.transforms(images)

        if torch.isnan(images).any():
            raise ValueError(f'NaN values found in transformed image for file {filename}')

        transformed_data = images.cpu().numpy().tobytes()
        if transformed_hash == "":
            transformed_hash = self.checksummer.compute_hash(transformed_data)
            self.transformed_hashes[idx] = transformed_hash

        return {
            'raw_hash': raw_hash,
            'transformed_hash': transformed_hash,
            'images': images,
            'label': treatment_code,
            'treatment': treatment,
        }

    def get_checksum(self, transformed: bool = False):
        if transformed:
            tag = 'dataset transformed'
        else:
            tag = 'dataset raw'

        return self.checksummer.compute_checksum(self, self.indices, tag, transformed_data=transformed)


    def get_random_sample(self, sample_size):
        sample_indices = random.sample(range(len(self)), sample_size)
        sample = [self[i] for i in sample_indices]
        if isinstance(sample[0]['images'][0], torch.Tensor):
            return torch.stack([x['images'][0] for x in sample], dim=0)
        else:
            return sample

    def ensure_ToTensor(self, composed_transforms):
        transform_list = composed_transforms.transforms
        if not any(isinstance(t, T.ToTensor) for t in transform_list):
            transform_list.append(T.ToTensor())
        return T.Compose(transform_list)
