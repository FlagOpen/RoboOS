import cv2
import numpy as np
import open3d as o3d
from IPython import embed
from PIL import Image
from sg.utils.utils import RECORDER_DIR, end2cam
from sg.utils.utils import pose_Euler_to_T
from sg.utils.utils import DEPTH_MIN, DEPTH_MAX, DEPTH_SCALE
from sg.utils.utils import transform_to_translation_quaternion
import json
import os
from tqdm import tqdm
import torch
import torchvision.transforms.functional as V
from torch import Tensor
from typing import NamedTuple
import shutil
import matplotlib.pylab as plt
import argparse

def apply_pose_to_xyz(xyz: np.ndarray, pose: np.ndarray) -> np.ndarray:
    xyz_reshaped = xyz.reshape(-1, 3)
    xyz_reshaped = np.hstack([xyz_reshaped, np.ones((xyz_reshaped.shape[0], 1))])
    xyz_transformed = np.dot(pose, xyz_reshaped.T).T[..., :3].reshape(xyz.shape)
    return xyz_transformed

def get_xyz(depth, intrinsic):
    fx, fy, cx, cy = intrinsic
    xmap, ymap = np.arange(depth.shape[1]), np.arange(depth.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depth
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z
    points_xyz = np.stack((points_x, points_y, points_z), axis=-1).astype(np.float32)
    return points_xyz

def rgbd_to_pcd(rgb_image_path, depth_image_path, pose_path, intrinsic):
    image = np.asarray(Image.open(rgb_image_path), dtype=np.uint8)
    depth = (np.asarray(Image.open(depth_image_path), dtype=np.uint16)) / DEPTH_SCALE
    T_cam_in_base = np.loadtxt(pose_path)
    # mask = np.logical_or(depth < DEPTH_MIN, depth > DEPTH_MAX)
    mask = np.logical_and(depth > DEPTH_MIN, depth < DEPTH_MAX)
    # mask = np.zeros_like(depth).astype(np.bool_)
    img = torch.from_numpy(image).permute(2, 0, 1)
    img = V.convert_image_dtype(img, torch.float32)
    rgb = img.permute(1, 2, 0).detach().cpu().numpy()
    xyz = get_xyz(depth, intrinsic)
    xyz= apply_pose_to_xyz(xyz, T_cam_in_base)
    rgb, xyz = rgb[mask], xyz[mask]

    import copy
    image_new = copy.deepcopy(image)
    image_new[~mask] = np.array([1, 1, 1])
    image_new = Image.fromarray(image_new)
    # image_new.save(os.path.basename(rgb_image_path))

    return rgb, xyz

def pose_process(pose_end_in_base):
    T_end_in_base = pose_Euler_to_T(pose_end_in_base).astype(np.float32)
    T_cam_in_base = end2cam(T_end_in_base).astype(np.float32)

    print(f"end: {transform_to_translation_quaternion(T_end_in_base)}")
    print(f"cam: {transform_to_translation_quaternion(T_cam_in_base)}")
    print("*******************************************")
    return T_cam_in_base

def rgb_depth_to_pcd():
    tags = "test"
    rgb_image_dir = RECORDER_DIR / tags / "rgb"
    depth_image_dir = RECORDER_DIR / tags / "depth"
    poses_dir = RECORDER_DIR / tags / "poses"
    intrinsic_path = RECORDER_DIR / tags / "calib.txt"
    intrinsic = np.loadtxt(intrinsic_path)

    color_image_paths = [os.path.join(rgb_image_dir, f) for f in 
                            sorted(os.listdir(rgb_image_dir), key=lambda x: int(os.path.basename(x).split(".")[0]))]
    depth_image_paths = [os.path.join(depth_image_dir, f) for f in 
                            sorted(os.listdir(depth_image_dir), key=lambda x: int(os.path.basename(x).split(".")[0]))]
    pose_paths = [os.path.join(poses_dir, f) for f in 
                            sorted(os.listdir(poses_dir), key=lambda x: int(os.path.basename(x).split(".")[0]))]

    xyzs = []
    rgbs = []
    for index in tqdm(range(0, len(color_image_paths), 5), desc="point cloud"):
        img_path = color_image_paths[index]
        depth_path = depth_image_paths[index]
        pose_path = pose_paths[index]
        name = os.path.basename(img_path)
        rgb, xyz = rgbd_to_pcd(img_path, depth_path, pose_path, intrinsic)
        rgbs.append(rgb)
        xyzs.append(xyz)

    xyzs = np.vstack(xyzs)
    rgbs = np.vstack(rgbs)

    # if T_world_in_base is not None:
    #     num_points = xyzs.shape[0]
    #     xyzs = np.hstack((xyzs, np.ones((num_points, 1))))
    #     xyzs = (T_world_in_base @ xyzs.T).T
    #     xyzs = xyzs[:, :3]

    merged_pcd = o3d.geometry.PointCloud()
    merged_pcd.points = o3d.utility.Vector3dVector(xyzs)
    merged_pcd.colors = o3d.utility.Vector3dVector(rgbs)

    # Flip the pcd
    # merged_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(visible=True)
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
    visualizer.add_geometry(merged_pcd)
    visualizer.add_geometry(coordinate_frame)
    visualizer.run()


if __name__ == "__main__":
    rgb_depth_to_pcd()