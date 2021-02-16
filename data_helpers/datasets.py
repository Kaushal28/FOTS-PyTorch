import os

import torch
from torch.utils.data import Dataset

import cv2

import numpy as np
import pandas as pd

import scipy.io as sio

from data_utils import generate_rbbox

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
    
    @staticmethod
    def _resize_image(image, image_size):
        """
        Resize the given image to image_size * image_size
        shaped square image.
        """
        # First pad the given image to match the image_size or image's larger
        # side (whichever is larger). [Create a square image]
        img_h, img_w, _ = image.shape
        max_size = max(image_size, img_w, img_h)

        # Create new square image of appropriate size
        img_padded = np.zeros((max_size, max_size, 3), dtype=np.float32)
        # Copy the original image into new image
        # (basically, new image is padded version of original image).
        img_padded[:img_h, :img_w, :] = image.copy()
        img_h, img_w, _ = img_padded.shape

        # if image_size higher that image sides, then the current padded
        # image will be of size image_size * image_size. But if not, resize the
        # padded iamge. This is done to keep the aspect ratio same even after
        # square resize.
        img_padded = cv2.resize(img_padded, dsize=(image_size, image_size))

        # We need the ratio of resized image width and heights to its
        # older dimensions to scale the bounding boxes accordingly
        scale_x = image_size / img_w
        scale_y = image_size / img_h

        return img_padded, scale_x, scale_y
    
    def _load_from_file(self, image_path, gt_path):
        """
        Load the image and corresponding ground truth from
        the file using given paths.
        """
        # Load the image
        image = cv2.imread(os.path.join(self.image_dir, image_path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0  # Normalize

        # Resize the image to required size
        image, scale_x, scale_y = self._resize_image(image, self.image_size)

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
        # would have changed
        bboxes[:, :, 0] *= scale_x  # scale x coordinate
        bboxes[:, :, 1] *= scale_y  # scale y coordinate
        
        return image, bboxes, transcript
