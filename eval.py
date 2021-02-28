import os
import json
import argparse

import cv2

import torch

import numpy as np

from data_helpers.data_utils import resize_image
from model import FOTSModel
from bbox import Toolbox

from pathlib import Path


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def _load_model(model_path):
    """Load model from given path to available device."""
    model = FOTSModel()
    model.to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE)["model"])
    return model


def _load_image(image_path):
    """Load image form given path."""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image, _, _ = resize_image(image, 512)

    image = torch.from_numpy(image[np.newaxis, :, :, :]).permute(0, 3, 1, 2)
    image = image.to(DEVICE)
    return image


def inference(args):
    """FOTS Inference on give images."""
    model = _load_model(args.model)
    for image_path in os.listdir(args.input_dir):
        with torch.no_grad():
            pred_bboxes, pred_transcripts = Toolbox.predict(Path(args.input_dir + os.sep + image_path), model, True, Path('.'), True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", type=str, required=True,
        help='Path to trained model'
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, default="predictions",
        help="Output directory to save predictions"
    )
    parser.add_argument(
        "-i", "--input_dir", type=str, required=True,
        help="Input directory having images to be predicted"
    )
    args = parser.parse_args()
    inference(args)