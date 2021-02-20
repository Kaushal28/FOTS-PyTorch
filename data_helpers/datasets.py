import os
from pathlib import Path

import torch
from torch.utils.data import Dataset

import cv2

import numpy as np
import pandas as pd
import scipy.io

from data_helpers.data_utils import generate_rbbox, resize_image

# TODO: Training masks and ignore bboxes having ### for training
class ICDARDataset(Dataset):
    """
    Define ICDAR Dataset based on its format.
    """

    def __init__(self, image_dir, gt_dir, image_size=512):
        """Constructor."""
        super().__init__()

        self.image_dir = image_dir
        self.gt_dir = gt_dir
        self.image_size = image_size

        if (
            not self.image_dir or
            not self.gt_dir or
            not os.path.isdir(self.image_dir) or
            not os.path.isdir(self.gt_dir)
        ):
            raise ValueError("Images and ground truths should be in separate dir.")

        images = os.listdir(self.image_dir)
        gts = os.listdir(self.gt_dir)

        if len(images) != len(gts):
            raise ValueError(
                f"Found inconsistent data: # of images: {len(images)} and # of ground truths: {len(gts)}"
            )

        data_paths = {}    
        for image in images:
            # Get image index from its file name
            idx = image.split('.')[0].split('_')[1]
            # Get corresponding ground truth
            data_paths[image] = f"gt_img_{idx}.txt"
        
        self.data_df = pd.DataFrame({
            "image_path": list(data_paths.keys()),
            "gt_path": list(data_paths.values()),
        })

    def __len__(self):
        """Define the length of the dataset."""
        return len(self.data_df)

    def __getitem__(self, index):
        """
        Retrieve the image and corresponding label at given index.
        
        The ground truth generation is based on this paper: (EAST)
        https://arxiv.org/pdf/1704.03155.pdf (Section 3.3 Label Generation).

        The first step is to shrink the given bbox by calculating
        reference lengths for each bbox vertex. The shrunk version
        of original bbox will be the actual bbox ground truth.

        Then a rotated rectangle is generated that covers the original bbox
        with minimal area. The angle of this rotated rectangle is considered
        as ground truths.
        """
        data = self.data_df.iloc[index]
        image_path, gt_path = data["image_path"], data["gt_path"]

        # Get raw input image and corresponding labels
        image, bboxes, transcripts = self._load_from_file(image_path, gt_path)

        # Extract score map (per pixel gt), pixel location map
        # (distance of the pixel from top, bottom, left and right sides of bbox)
        # and bbox angle map. These are required by text detection branch of FOTS
        # shape of score map: (img_size/4 * img_size/4 * 1)

        # Get pixel location/geography map
        # shape of geo_map: (img_size/4 * img_size/4 * 5)
        score_map, geo_map, bboxes = generate_rbbox(image, bboxes, transcripts)

        return image_path, image, bboxes.reshape(-1, 8), transcripts, score_map, geo_map
    
    def _load_from_file(self, image_path, gt_path):
        """
        Load the image and corresponding ground truth from
        the file using given paths.
        """
        # Load the image
        image = cv2.imread(os.path.join(self.image_dir, image_path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # image /= 255.0  # Normalize

        # Resize the image to required size
        image, scale_x, scale_y = resize_image(image, self.image_size)

        # Extract ground truth bboxes
        # Reference: https://stackoverflow.com/a/49150749/5353128
        with open(os.path.join(self.gt_dir, gt_path), 'r', encoding='utf-8-sig') as file:
            content = file.read().split('\n')
            # Removing empty lines (possibly the last line in CSV)
            content = [line for line in content if line]

            # Extract bboxes and convert them to numpy array of size n_box * 4 * 2
            # where 4 is four coordinates of rectangle and 2 is for x and y components
            # of each coordinate
            bboxes = list(map(lambda str: str.split(',')[:-1], content))
            bboxes = np.array([np.array(bbox)[:8].reshape(4, 2) for bbox in bboxes]).astype(np.float32)
        
            transcript = np.array(
                list(map(lambda str: str.split(',')[-1], content)), dtype='object'
            )
        
        # Scale the bounding boxes as per the resized image
        # This is required because after resize, the position of the texts
        # would have changed. Shape of bboxes: n_words * 4 * 2
        bboxes[:, :, 0] *= scale_x  # scale x coordinate
        bboxes[:, :, 1] *= scale_y  # scale y coordinate

        return image, bboxes, transcript


class Synth800kDataset(Dataset):
    """
    Define Synth800K Dataset based on its format.
    """

    def __init__(self, image_dir, gt_path, image_size=512):
        """Constructor."""
        super().__init__()

        self.image_dir = image_dir  # Path of the dir where all the images are stored
        self.gt_path = gt_path  # Path of gt.mat object
        self.image_size = image_size  # Size of images for training

        if (
            not self.image_dir or
            not self.gt_path or
            not os.path.isdir(self.image_dir) or
            not os.path.isfile(self.gt_path)
        ):
            raise ValueError("Some of the parameter(s) is/are invalid.")

        # Load the ground truth matrix/object
        # Reference: https://www.robots.ox.ac.uk/~vgg/data/scenetext/readme.txt
        mat = scipy.io.loadmat(self.gt_path)

        # Convert to dataframe for ease of operations
        self.data_df = pd.DataFrame({
            'imnames': np.concatenate(mat['imnames'][0], axis=0),
            'wordBB': np.concatenate(mat['wordBB'], axis=0),
            'txt': np.concatenate(mat['txt'], axis=0)
        })

        # Clean up
        del mat

        # Filter out the gt dataframe if training on subset of images
        image_ids = list(map(str, list(Path(self.image_dir).rglob('*.jpg'))))

        # Append the image dir (base dir) name in imnames
        self.data_df['imnames'] = self.image_dir + os.sep + self.data_df['imnames'].astype(str)
        self.data_df = self.data_df[self.data_df['imnames'].isin(image_ids)].reset_index(drop=True)

    def __len__(self):
        """Define the length of the dataset."""
        return len(self.data_df)
    
    def __getitem__(self, index):
        """
        Retrieve the image and corresponding label at given index.
        
        The ground truth generation is based on this paper: (EAST)
        https://arxiv.org/pdf/1704.03155.pdf (Section 3.3 Label Generation).

        The first step is to shrink the given bbox by calculating
        reference lengths for each bbox vertex. The shrunk version
        of original bbox will be the actual bbox ground truth.

        Then a rotated rectangle is generated that covers the original bbox
        with minimal area. The angle of this rotated rectangle is considered
        as ground truths.
        """
        data = self.data_df.iloc[index]
        image_path, bboxes, transcripts = data["imnames"], data["wordBB"], data["txt"]

        # Get raw input image (resized)
        image, bboxes = self._load_from_file(image_path, bboxes)

        transcripts = list(map(str.strip, transcripts))  # TODO: Filter characters which are not in classes.

        # Extract score map (per pixel gt), pixel location map
        # (distance of the pixel from top, bottom, left and right sides of bbox)
        # and bbox angle map. These are required by text detection branch of FOTS
        # shape of score map: (img_size/4 * img_size/4 * 1)

        # Get pixel location/geography map
        # shape of geo_map: (img_size/4 * img_size/4 * 5)
        score_map, geo_map, training_mask, bboxes = generate_rbbox(image, bboxes, transcripts)

        return image_path, image, bboxes.reshape(-1, 8), training_mask, transcripts, score_map, geo_map
    
    def _load_from_file(self, image_path, bboxes):
        """Load the image from the file using given path."""
        # Load the image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # image /= 255.0  # Normalize

        # Resize the image to required size
        image, scale_x, scale_y = resize_image(image, self.image_size)

        # Scale the bounding boxes as per the resized image
        # This is required because after resize, the position of the texts
        # would have changed. Bboxes shape: 2 * 4 * n_words
        if len(bboxes.shape) < 3:
            bboxes = bboxes[:, :, np.newaxis]
        bboxes[0, :, :] *= scale_x  # scale x coordinate
        bboxes[1, :, :] *= scale_y  # scale y coordinate
        
        bboxes = np.moveaxis(bboxes, [0, 2], [2, 0])
        return image, bboxes


class Synth800kPreprocessedDataset(Dataset):
    def __init__(self, df):
        self.df = df
    
    def __getitem__(self, index):
        data = self.df.iloc[index]
        image = np.load(f'../input/synth800kpreprocessed/{data["images"]}').astype(np.float32)
        score_map = np.load(f'../input/synth800kpreprocessed/{data["score_maps"]}').astype(np.float32)
        geo_map = np.load(f'../input/synth800kpreprocessed/{data["geo_maps"]}').astype(np.float32)
        training_mask = np.load(f'../input/synth800kpreprocessed/{data["training_masks"]}').astype(np.float32)
        return torch.from_numpy(image), torch.from_numpy(score_map), torch.from_numpy(geo_map), training_mask
    
    def __len__(self):
        return len(self.df)
