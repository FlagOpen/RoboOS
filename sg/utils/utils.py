import numpy as np
import shutil
import os
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import open3d as o3d
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from typing import Union

TIMEINTERVAL = 0.02

CAM_IN_END_Translation = [-0.01131290, 0.05520051, 0.10699229]  # px, py, pz
CAM_IN_END_Quaternion = [0.00685809, -0.00771620, 0.99855130, 0.05280834]  # qx, qy, qz, qw


T_cam_in_end = np.eye(4)
T_cam_in_end[:3, :3] = R.from_quat(CAM_IN_END_Quaternion).as_matrix()
T_cam_in_end[:3, 3] = CAM_IN_END_Translation


#WH = [1280, 720]  
#WH = [1024, 768]  
WH = [640, 480]

DEPTH_MIN = 0  # 0.2
DEPTH_MAX = 2  # 2

DEPTH_SCALE = 1000
FPS = 30

AGV_IP = "192.168.188.168"
AGV_PORT = 9090
ROT2AGV_OFFSET_X = 0.11
ROT2AGV_ROTATION_ANGLE = 135
AVG_RATE_HZ = 10


CROP_INFO = {
    "top": 100,
    "bottom": 400,
    "left": 50,
    "right": 600
}

RECORDER_DIR = Path("data_example")

# sam2
sam2_checkpoint_path = "checkpoints/segment-anything-2/sam2_hiera_large.pt"
sam2_model_cfg_path = "../sam2_configs/sam2_hiera_l.yaml"

# groundingdino
grounding_dino_config_path = "checkpoints/GroundingDINO/GroundingDINO_SwinT_OGC.py"
grounding_dino_checkpoint_path = "checkpoints/GroundingDINO/groundingdino_swint_ogc.pth"

# ram
ram_checkpoint_path = "checkpoints/recognize_anything/ram_swin_large_14m.pth"

# clip
clip_model_name = "ViT-H-14"
clip_checkpoint_path = "checkpoints/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin"

# anygrasp
anygrasp_checkpoint_path = "checkpoints/anygrasp/checkpoint_detection.tar"

#
bert_base_uncased_path = "checkpoints/bert-base-uncased"

# ROBOT_IP = "192.168.188.6"
# LOOP_SCENE_ID = "10177"
# RECOVERY_SCENE_ID = "10173"

# ROBOT_IP = "192.168.188.5"
# LOOP_SCENE_ID = "10173"
# RECOVERY_SCENE_ID = "10174"
# SEE_SCENE_ID = "10175"
# SEE_SCENE_ID = "10174"

def clean_dir(dir_list: list):
    for dirname in dir_list:
        if os.path.exists(dirname):
            shutil.rmtree(dirname)
        os.makedirs(dirname, exist_ok=True)


def run_scene(robot, SCENE_ID, is_wait=False):
    r_task_id = robot.start_task(SCENE_ID)
    if is_wait:
        robot.wait_task(r_task_id)

def pose_Euler_to_T(actual_flange_pose: dict): # ZYX
    px = actual_flange_pose['x']
    py = actual_flange_pose['y']
    pz = actual_flange_pose['z']
    rx = actual_flange_pose['rx']
    ry = actual_flange_pose['ry']
    rz = actual_flange_pose['rz']

    position = np.array([px, py, pz])
    orientation_matrix_ZYX = R.from_euler('ZYX', [rz, ry, rx]).as_matrix()

    T_zyx = np.eye(4)
    T_zyx[:3, :3] = orientation_matrix_ZYX
    T_zyx[:3, 3] = position
    return T_zyx.astype(np.float32)

def pose_T_to_Euler(T_zyx):
    position = T_zyx[:3, 3]
    orientation_matrix_ZYX = T_zyx[:3, :3]
    euler_angles_zyx = R.from_matrix(orientation_matrix_ZYX).as_euler('ZYX')

    pose_dict = {
        'x': str(position[0]),
        'y': str(position[1]),
        'z': str(position[2]),
        'rx': str(euler_angles_zyx[2]),
        'ry': str(euler_angles_zyx[1]),
        'rz': str(euler_angles_zyx[0])
    }
    return pose_dict
    
def end2cam(T_end_in_base):
    T_cam_in_base = np.dot(T_end_in_base, T_cam_in_end)
    return T_cam_in_base

def transform_to_translation_quaternion(transform: np.ndarray) -> tuple:
    translation = transform[:3, 3]
    rotation_matrix = transform[:3, :3]
    quaternion = R.from_matrix(rotation_matrix).as_quat()
    return translation, quaternion

def translation_quaternion_to_transform(translation: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    rotation_matrix = R.from_quat(quaternion).as_matrix()
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = translation
    return transform


def get_inlier_mask(point, color, mask, nb_neighbors: int=30, std_ratio: float=1.5):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point[mask])
    pcd.colors = o3d.utility.Vector3dVector(color[mask])
    _, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    inlier_mask = np.zeros_like(mask).flatten()
    mask_valid = np.where(mask.flatten() > 0)[0]
    inlier_mask[mask_valid[ind]] = 1
    return inlier_mask.reshape(mask.shape).astype(bool)



def depth_to_color_vis(recorder_dir: Path, top: Union[int, None]=None):
    depth_dir = recorder_dir / "depth"
    color_dir = recorder_dir / "rgb"
    depth_color_dir = recorder_dir / "depth_color"
    depth_color_dir.mkdir(parents=True, exist_ok=True)
    
    depth_filepaths = [color_dir / f for f in 
                        sorted(os.listdir(color_dir), key=lambda x: int(x.split(".")[0]))]
    color_filepaths = [color_dir / f for f in 
                        sorted(os.listdir(color_dir), key=lambda x: int(x.split(".")[0]))]
    
    assert len(depth_filepaths) == len(color_filepaths)

    path_len = len(color_filepaths) if top is not None else top

    # for cnt in tqdm(range(len(depth_filepaths)), desc="Save depth and rgb Images."):
    for cnt in tqdm(range(10), desc="Save depth and rgb Images."):
        color_path = color_filepaths[cnt]
        depth_path = depth_filepaths[cnt]

        rgb = cv2.imread(color_path, cv2.IMREAD_COLOR)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        
        depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_JET)

        alpha = 0.8
        overlay_image = cv2.addWeighted(rgb, alpha, depth_colored, 1 - alpha, 0)

        plt.figure(figsize=(10, 10))
        plt.subplot(1, 3, 1)
        plt.title('RGB Image')
        plt.imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))

        plt.subplot(1, 3, 2)
        plt.title('Depth Image')
        plt.imshow(depth_colored, cmap='jet')  # 确保显示的颜色映射与上面一致

        plt.subplot(1, 3, 3)
        plt.title('Overlay Image')
        plt.imshow(cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB))

        plt.savefig(depth_color_dir / (color_path.stem + ".png"), bbox_inches="tight")
        plt.close()

def vis_depth(recorder_dir: Path, top_k: Union[int, None]=10):
    depth_dir = recorder_dir / "depth"
    depth_color_dir = recorder_dir / "vis_depth"
    depth_color_dir.mkdir(parents=True, exist_ok=True)
    depth_filepaths = [depth_dir / f for f in 
                        sorted(os.listdir(depth_dir), key=lambda x: int(x.split(".")[0]))]
    
    path_len = len(depth_filepaths) if top_k is None else top_k
    for cnt in tqdm(range(path_len), desc="Save depth Images."):
        depth_path = depth_filepaths[cnt]
        depth = np.load(depth_path)
        
        depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        
        depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_JET)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(depth_colored, cmap='jet')
        plt.axis('off')

        plt.savefig(depth_color_dir / (depth_path.stem + ".png"), bbox_inches="tight", pad_inches=0)
        plt.close()