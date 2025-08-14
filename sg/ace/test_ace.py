#!/usr/bin/env python3
# Copyright © Niantic, Inc. 2022.

import logging
from distutils.util import strtobool
from pathlib import Path
import numpy as np
import torch
from torch.cuda.amp import autocast
import dsacstar
from ace.ace_network import Regressor
from tqdm import tqdm
import torchvision.transforms.functional as TF
import numpy as np
from torchvision import transforms
from omegaconf import OmegaConf

class AceTestdataProcess:
    def __init__(self, use_half=True, image_height=384, augment=False, aug_black_white=0.1):
        self.use_half = use_half
        self.image_height = image_height
        self.augment = augment
        self.aug_black_white = aug_black_white
        
        if self.augment:
            self.image_transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.ColorJitter(brightness=self.aug_black_white, contrast=self.aug_black_white),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4],
                    std=[0.25]
                ),
            ])
        else:
            self.image_transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4],
                    std=[0.25]
                ),
            ])
    
    @staticmethod
    def _resize_image(image, image_height):
        image = TF.to_pil_image(image)
        image = TF.resize(image, image_height)
        return image
    
    def process_frames(self, observations):
        batch_data = []
        for name, obs in observations.items():
            rgb = obs["rgb"]
            depth = obs["depth"]
            mask = obs["mask"]
            calib = obs["intrinsic"]
            
            # Resize image
            scale_factor = self.image_height / rgb.shape[0]
            rgb = self._resize_image(rgb, self.image_height)
            
            # Create image mask
            image_mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
            
            # Apply transformations
            image = self.image_transform(rgb)
            
            # Convert depth to meters
            depth = depth.astype(np.float64) / 1000.0
            
            # Create intrinsics matrix and its inverse
            intrinsics = torch.eye(3)
            intrinsics[0, 0] = calib[0, 0] * scale_factor
            intrinsics[1, 1] = calib[1, 1] * scale_factor
            intrinsics[0, 2] = calib[0, 2] * scale_factor
            intrinsics[1, 2] = calib[1, 2] * scale_factor
            
            intrinsics_inv = torch.inverse(intrinsics)
            
            # Convert to half precision if needed
            if self.use_half and torch.cuda.is_available():
                image = image.half()
            
            # Binarize the mask
            image_mask = image_mask > 0
            
            # Create the output dictionary
            out = {
                'image': image.unsqueeze(0),
                'image_mask': image_mask.unsqueeze(0),
                'intrinsics': intrinsics.unsqueeze(0),
                'intrinsics_inv': intrinsics_inv.unsqueeze(0),
                'scene_coords': torch.tensor(depth).unsqueeze(0)  # Assuming depth is used for scene_coords
            }
            
            batch_data.append(out)
        
        return batch_data


def test_ace(network, observations):
    # Setup logging.
    logging.basicConfig(level=logging.INFO)

    options_path = "ace/configs/test.yml"
    options = OmegaConf.load(options_path)

    options.network = network

    device = torch.device("cuda")
    head_network_path = Path(options.network)
    encoder_path = Path(options.encoder_path)
    dataprocess = AceTestdataProcess(
        use_half=False, 
        image_height=options.image_resolution, 
        augment=False, 
        aug_black_white=0.1
    )

    # Setup dataloader. Batch size 1 by default.
    # testset_loader = DataLoader(testset, shuffle=False, num_workers=6)
    # Load network weights.
    encoder_state_dict = torch.load(encoder_path, map_location="cpu")
    head_state_dict = torch.load(head_network_path, map_location="cpu")

    # Create regressor.
    network = Regressor.create_from_split_state_dict(encoder_state_dict, head_state_dict)

    # Setup for evaluation.
    network = network.to(device)
    network.eval()

    # Testing loop.
    estimation_poses = []
    with torch.no_grad():
        frame_datas = dataprocess.process_frames(observations=observations)
        for frame_data in frame_datas:
            image_B1HW = frame_data['image']
            intrinsics_B33 = frame_data['intrinsics']
            image_B1HW = image_B1HW.to(device, non_blocking=True)

            # Predict scene coordinates.
            with autocast(enabled=True):
                scene_coordinates_B3HW = network(image_B1HW)

            # We need them on the CPU to run RANSAC.
            scene_coordinates_B3HW = scene_coordinates_B3HW.float().cpu()

            # Each frame is processed independently.
            scene_coordinates_3HW, intrinsics_33 = scene_coordinates_B3HW[0], intrinsics_B33[0]

            # Extract focal length and principal point from the intrinsics matrix.
            focal_length = intrinsics_33[0, 0].item()
            ppX = intrinsics_33[0, 2].item()
            ppY = intrinsics_33[1, 2].item()
            # We support a single focal length.
            # assert torch.allclose(intrinsics_33[0, 0], intrinsics_33[1, 1])
            # Allocate output variable.
            out_pose = torch.zeros((4, 4))

            # Compute the pose via RANSAC.
            inlierMap_num = dsacstar.forward_rgb(
                scene_coordinates_3HW.unsqueeze(0),
                out_pose,
                options.hypotheses, # 64
                options.threshold, # 10
                focal_length,
                ppX,
                ppY,
                options.inlieralpha, # 100
                options.maxpixelerror, # 100
                network.OUTPUT_SUBSAMPLE,
            )
            # print(out_pose)
            estimation_poses.append([out_pose.cpu().numpy(), inlierMap_num])
            
    del network
    torch.cuda.empty_cache()
    return estimation_poses