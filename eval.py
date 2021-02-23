import os
import json
import argparse

import cv2

import torch

import numpy as np

from data_helpers.data_utils import resize_image
from model import FOTSModel
from bbox import Toolbox

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def _load_model(model_path):
    """Load model from given path to available device."""
    model = FOTSModel()
    model.to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
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
        image = _load_image(os.path.join(args.input_dir, image_path))

        # Forward pass
        pred_score_map, pred_geo_map = model(image)

        pred_score_map = pred_score_map.permute(0, 2, 3, 1).detach().cpu().numpy()
        pred_geo_map = pred_geo_map.permute(0, 2, 3, 1).detach().cpu().numpy()

        pred_bboxes = []
        for idx in range(pred_score_map.shape[0]): 
            bboxes = Toolbox.detect(
                score_map=pred_score_map[idx, :, :, 0],
                geo_map=pred_geo_map[idx, :, :, ]
            )
            if len(bboxes) > 0:
                pred_bboxes.append(bboxes)
        
        pred_bboxes = np.concatenate(pred_bboxes)

        image = image.permute(0, 2, 3, 1)[0].cpu().detach().numpy()

        for i in range(pred_bboxes.shape[0]):
            # Define predicted rectangle vertices
            vertices = [
                [pred_bboxes[i][0], pred_bboxes[i][1]],
                [pred_bboxes[i][2], pred_bboxes[i][3]],
                [pred_bboxes[i][4], pred_bboxes[i][5]],
                [pred_bboxes[i][6], pred_bboxes[i][7]]
            ]
            cv2.polylines(image, [np.array(vertices).astype(np.int32)], isClosed=True, color=(255, 255, 0), thickness=1)
        
        # Save the image
        cv2.imwrite(os.path.join(args.output_dir, os.path.basename(image_path)), image)


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