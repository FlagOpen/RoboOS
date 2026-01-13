import cv2
import numpy as np
import open3d as o3d
from PIL import Image
from sg.utils.utils import RECORDER_DIR, end2cam
from sg.utils.utils import pose_Euler_to_T
from sg.utils.utils import transform_to_translation_quaternion
# from utils.utils import RECORDER_DIR, end2cam
# from utils.utils import pose_Euler_to_T
# from utils.utils import transform_to_translation_quaternion
import os
from tqdm import tqdm
import torch
import torchvision.transforms.functional as V
import argparse

def apply_pose_to_xyz(xyz: np.ndarray, pose: np.ndarray) -> np.ndarray:
    # xyz_reshaped = xyz.reshape(-1, 3)
    # xyz_reshaped = np.hstack([xyz_reshaped, np.ones((xyz_reshaped.shape[0], 1))])
    # xyz_transformed = np.dot(xyz_reshaped, pose.T)[..., :3].reshape(xyz.shape)
    # return xyz_transformed
    _xyz = xyz.reshape(-1, 3)
    xyz_new = (_xyz @ pose[:3, :3].T + pose[:3, 3]).reshape(xyz.shape)
    return xyz_new

def get_xyz(depth, intrinsic):
    fx, fy, cx, cy = intrinsic
    xmap, ymap = np.arange(depth.shape[1]), np.arange(depth.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depth
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z
    points_xyz = np.stack((points_x, points_y, points_z), axis=-1).astype(np.float32)
    return points_xyz

def get_pcd(rgb_image_path, point_path, mask_path, pose_path, get_inlier_mask=None):
    image = np.asarray(Image.open(rgb_image_path), dtype=np.uint8)
    T_cam_in_base = np.loadtxt(pose_path)
    xyz = np.load(point_path)
    mask = np.load(mask_path)
    rgb = image / 255
    # img = torch.from_numpy(image).permute(2, 0, 1)
    # img = V.convert_image_dtype(img, torch.float32)
    # rgb = img.permute(1, 2, 0).detach().cpu().numpy()
    if get_inlier_mask is not None:
        inlier_mask = get_inlier_mask(xyz, rgb)
        mask = mask * np.bool_(inlier_mask)
    xyz= apply_pose_to_xyz(xyz, T_cam_in_base)
    rgb, xyz = rgb[mask], xyz[mask]
    return rgb, xyz

def pose_process(pose_end_in_base):
    T_end_in_base = pose_Euler_to_T(pose_end_in_base).astype(np.float32)
    T_cam_in_base = end2cam(T_end_in_base).astype(np.float32)

    print(f"end: {transform_to_translation_quaternion(T_end_in_base)}")
    print(f"cam: {transform_to_translation_quaternion(T_cam_in_base)}")
    print("*******************************************")
    return T_cam_in_base

def rgb_depth_to_pcd(args):
    tags = args.tags
    rgb_image_dir = RECORDER_DIR / tags / "rgb"
    point_image_dir = RECORDER_DIR / tags / "point"
    poses_dir = RECORDER_DIR / tags / args.pose_tags
    mask_dir = RECORDER_DIR / tags / "mask"
    
    color_image_paths = [os.path.join(rgb_image_dir, f) for f in 
                            sorted(os.listdir(rgb_image_dir), key=lambda x: int(os.path.basename(x).split(".")[0]))]
    point_paths = [os.path.join(point_image_dir, f) for f in 
                            sorted(os.listdir(point_image_dir), key=lambda x: int(os.path.basename(x).split(".")[0]))]
    pose_paths = [os.path.join(poses_dir, f) for f in 
                            sorted(os.listdir(poses_dir), key=lambda x: int(os.path.basename(x).split(".")[0]))]
    mask_paths = [os.path.join(mask_dir, f) for f in 
                            sorted(os.listdir(mask_dir), key=lambda x: int(os.path.basename(x).split(".")[0]))]

    xyzs = []
    rgbs = []
    for index in tqdm(range(0, len(color_image_paths), 5), desc="point cloud"):
        img_path = color_image_paths[index]
        point_path = point_paths[index]
        pose_path = pose_paths[index]
        mask_path = mask_paths[index]

        rgb, xyz = get_pcd(img_path, point_path, mask_path, pose_path)
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

    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([merged_pcd, coordinate_frame])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Show PointCloud.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--tags', type=str, default="room1", help='tags for scene.')
    parser.add_argument('--pose_tags', type=str, default="poses_droidslam", help='tags for scene.')

    args = parser.parse_args()
    rgb_depth_to_pcd(args)