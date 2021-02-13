import os

import torch
from torch.utils.data import Dataset

import cv2

import numpy as np
import pandas as pd

import scipy.io as sio


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
        """Retrieve the image and corresponding label at given index."""
        data = self.data_df.iloc[index]
        image_path, gt_path = data["image_path"], data["gt_path"]

        # Get raw input image and corresponding labels
        image, bboxes, transcripts = self._load_from_file(image_path, gt_path)

        # Extract score map (per pixel gt), pixel location map
        # (distance of the pixel from top, bottom, left and right sides of bbox)
        # and bbox angle map. These are required by text detection branch of FOTS
        # shape of score map: (img_size/4 * img_size/4 * 1)
        score_map = self._get_score_map(image, bboxes)

        # Get pixel location/geography map
        # shape of geo_map: (img_size/4 * img_size/4 * 5)
        geo_map = self._get_geo_map(image, bboxes)

        return image_path, image, bboxes, transcripts, score_map, geo_map

    def _get_geo_map(self, image, bboxes):
        """
        For each pixel in the text bbox, get the following 5 channels:
        distances to top, bottom, left, right sides of the bounding box that
        contains the pixel. The last channel contains the angle/orientation 
        info. for each bbox. This is called geography map.
        """
        img_height, img_width, _ = image.shape
        # First initialize the geomap with all zeros
        geo_map = np.zeros((img_height, img_width, 5), dtype = np.float32)

        # Helper bbox mask layer
        bbox_mask = np.zeros((img_height, img_width), dtype=np.uint8)

        # Iterate over each bbox and fill geo_map
        for idx, bbox in enumerate(bboxes):
            cv2.fillPoly(bbox_mask, [bbox.astype(np.int32)], idx+1)

            # Get all the bbox points for which the distance is to be
            # calculated.
            bbox_points = np.argwhere(bbox_mask == idx+1)

            # for each bbox point, calc. 4 distances and rotation and fill the
            # geo map.
            for bbox_y, bbox_x in bbox_points:
                bbox_point = np.array([bbox_x, bbox_y], dtype=np.float32)
                # distance from top
                geo_map[bbox_y, bbox_x, 0] = self._point_to_line_dist(bbox[0], bbox[1], bbox_point)
                # distance from right
                geo_map[bbox_y, bbox_x, 1] = self._point_to_line_dist(bbox[1], bbox[2], bbox_point)
                # distance from bottom
                geo_map[bbox_y, bbox_x, 2] = self._point_to_line_dist(bbox[2], bbox[3], bbox_point)
                # distance from left
                geo_map[bbox_y, bbox_x, 3] = self._point_to_line_dist(bbox[3], bbox[0], bbox_point)
                # # bbox rotation angle
                geo_map[bbox_y, bbox_x, 4] = self._bbox_angle(bbox)

        # Size of the feature map from shared convolutions is 1/4th of
        # original image size. So all this geo_map should be of the
        # same size.
        geo_map = geo_map[::4, ::4].astype(np.float32)
        return geo_map

    @staticmethod
    def _bbox_angle(bbox):
        """Find the angle of rotation of given bbox."""
        # Considering bottom edge of rectangle for determining the angle.
        p0, p1 = bbox[2], bbox[3]

        # Reference: https://stackoverflow.com/a/2676810/5353128
        delta_x = p0[0] - p1[0]
        delta_y = p1[1] - p0[1]
        theta = np.degrees(np.arctan2(delta_y, delta_x))

        # Keep the range of theta in -90 to 90
        if theta < 0:
            theta = 180 - np.abs(theta)
        
        if theta > 90:
            theta = theta - 180

        return theta

    
    @staticmethod
    def _point_to_line_dist(p1, p2, p3):
        """
        Find perpendicular distance from point p3 to line passing through
        p1 and p2.
        """
        # Reference: https://stackoverflow.com/a/39840218/5353128
        return np.linalg.norm(np.cross(p2-p1, p1-p3)) / np.linalg.norm(p2-p1)

    def _get_score_map(self, image, bboxes):
        """
        Extract score map for given image and corresponding bbox info.
        Score map represents whether the pixel is part of text bbox or not.
        """
        # First create empty score map same as the size of image
        img_height, img_width, _ = image.shape
        score_map = np.zeros((img_height, img_width), dtype=np.uint8)

        # Based on bboxes, fill the score map to 1. This will create 2 class
        # classification label which indicates whether the text exists for
        # each pixel.
        for bbox in bboxes:
            cv2.fillPoly(score_map, [bbox.astype(np.int32)], 1)
        
        # Size of the feature map from shared convolutions is 1/4th of
        # original image size. So all these label maps should be of the
        # same size.
        score_map = score_map[::4, ::4, np.newaxis].astype(np.float32)

        return score_map
    
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
            bboxes = np.array([np.array(bbox).reshape(4, 2) for bbox in bboxes]).astype(np.float32)
        
            transcript = np.array(
                list(map(lambda str: str.split(',')[-1], content)), dtype='object'
            )
        
        # Scale the bounding boxes as per the resized image
        # This is required because after resize, the position of the texts
        # would have changed
        bboxes[:, :, 0] *= scale_x  # scale x coordinate
        bboxes[:, :, 1] *= scale_y  # scale y coordinate
        
        return image, bboxes, transcript
