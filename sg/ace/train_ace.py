#!/usr/bin/env python3
# Copyright © Niantic, Inc. 2022.

import logging
from pathlib import Path
from omegaconf import OmegaConf
from ace.ace_trainer import TrainerACE
import torch


def train_ace(scene, output_map_file):
    # Setup logging levels.
    logging.basicConfig(level=logging.INFO)

    options_path = "ace/configs/train.yml"
    options = OmegaConf.load(options_path)
    
    options.scene = scene
    options.output_map_file = output_map_file

    trainer = TrainerACE(options)
    trainer.train()

    del trainer
    torch.cuda.empty_cache()

if __name__ == '__main__':
    scene = Path("data_example/company")
    output_map_file = scene / "ace/ace.pt"
    train_ace(scene, output_map_file)
