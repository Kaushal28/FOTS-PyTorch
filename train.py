import os
import json
import random
import argparse

import torch
from torch.utils.data import DataLoader
from torch import nn, optim

import numpy as np
import pandas as pd

# from eval_tools.icdar2015 import eval as icdar_eval

from trainer import Train
from data_helpers.datasets import ICDARDataset, Synth800kPreprocessedDataset
from data_helpers.data_utils import icdar_collate
from components.loss import FOTSLoss
from model import FOTSModel
from trainer import Train


def fots_metric(pred, gt):
    config = icdar_eval.default_evaluation_params()
    output = icdar_eval.eval(pred, gt, config)
    return output['method']['precision'], output['method']['recall'], output['method']['hmean']


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(config):
    """Main entry point of train module."""
    # Initialize the dataset
    # Full dataset
    # dataset = ICDARDataset('/content/ch4_training_images', '/content/ch4_training_localization_transcription_gt')
    data_df = pd.read_csv(f"{config['data_base_dir']}/gt/train.csv")
    dataset = Synth800kPreprocessedDataset(config["data_base_dir"], data_df)

    # Train test split
    val_size = config["val_fraction"]
    val_len = int(val_size * len(dataset))
    train_len = len(dataset) - val_len
    icdar_train_dataset, icdar_val_dataset = torch.utils.data.random_split(
        dataset, [train_len, val_len]
    )

    icdar_train_data_loader = DataLoader(
        icdar_train_dataset,
        pin_memory=True,
        **config["dataset_config"],
        worker_init_fn=seed_worker
        # collate_fn=icdar_collate
    )

    icdar_val_data_loader = DataLoader(
        icdar_val_dataset,
        **config["dataset_config"],
        pin_memory=True,
        worker_init_fn=seed_worker
        # collate_fn=icdar_collate
    )

    # Initialize the model
    model = FOTSModel()

    # Count trainable parameters
    print(f'The model has {count_parameters(model):,} trainable parameters.')

    loss = FOTSLoss()
    optimizer = model.get_optimizer(config["optimizer"], config["optimizer_config"])

    lr_schedular = getattr(
        optim.lr_scheduler, config["lr_schedular"], "ReduceLROnPlateau"
    )(optimizer, **config["lr_scheduler_config"])

    trainer = Train(
        model, icdar_train_data_loader, icdar_val_data_loader, loss,
        fots_metric, optimizer, lr_schedular, config
    )

    trainer.train()


def seed_all(seed=28):
    """Seed everything for result reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def seed_worker(_worker_id):
    """Seed data loader workers."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


if __name__ == '__main__':

    # First seed everything
    seed_all()

    # Parse command line args to get the config file
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config', default="../config/train_config.json",
        type=str, help='Training config file path.'
    )
    args = parser.parse_args()

    if args.config is not None:
        with open(args.config, "r") as f:
            config = json.load(f)
        main(config)
    else:
        print("Invalid training configuration file provided.")
