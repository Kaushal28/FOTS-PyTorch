import os
import json
import argparse

import numpy as np
import pandas as pd

from torch.utils.data import DataLoader

from datasets import Synth800kDataset
from data_utils import synth800k_collate

from tqdm import tqdm


def main(config):
    """Entry point."""
    dataset = Synth800kDataset(config["data_dir"])
    data_loader = DataLoader(
        dataset,
        num_workers=config["num_workers"],
        pin_memory=True,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=synth800k_collate
    )

    os.makedirs(os.path.join(config["output_dir"], "image"), exist_ok=True)
    os.makedirs(os.path.join(config["output_dir"], "score"), exist_ok=True)
    os.makedirs(os.path.join(config["output_dir"], "geo"), exist_ok=True)
    os.makedirs(os.path.join(config["output_dir"], "training_mask"), exist_ok=True)

    img_list, sm_list, gm_list, tm_list = [], [], [], []
    for idx, batch in tqdm(enumerate(data_loader)):
        image_paths, images, score_maps, geo_maps, training_masks = batch
        for pth, i, s, g, tm in zip(image_paths, images, score_maps, geo_maps, training_masks):
            img_pth = pth.split("/")[-2:]
            img_name = img_pth[-1].split(".")[0]

            img_list.append(f"{config['output_dir']}/image/{img_name}.npy")
            sm_list.append(f"{config['output_dir']}/score/{img_name}_score_map.npy")
            gm_list.append(f"{config['output_dir']}/geo/{img_name}_geo_map.npy")
            tm_list.append(f"{config['output_dir']}/training_mask/{img_name}_tm.npy")

            np.save(f"{config['output_dir']}/image/{img_name}.npy", i.numpy().astype(np.uint8))
            np.save(f"{config['output_dir']}/score/{img_name}_score_map.npy", s.numpy().astype(np.uint8))
            np.save(f"{config['output_dir']}/geo/{img_name}_geo_map.npy", g.numpy().astype(np.float32))
            np.save(f"{config['output_dir']}/training_mask/{img_name}_tm.npy", tm.numpy().astype(np.uint8))
        
        if idx == 100:
            break

    data_df = pd.DataFrame({
        "images": img_list,
        "score_maps": sm_list,
        "geo_maps": gm_list,
        "training_masks": tm_list
    })
    data_df.to_csv(f"{config['output_dir']}/train.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config', default="../config/data_preprocessor_config.json",
        type=str, help='Data preprocessor config file path.'
    )
    args = parser.parse_args()

    if args.config is not None:
        with open(args.config, "r") as f:
            config = json.load(f)
        main(config)
    else:
        print("Invalid configuration file provoded.")
