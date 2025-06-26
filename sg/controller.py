import numpy as np
from sg.utils.utils import RECORDER_DIR, get_inlier_mask
from sg.scripts.zmq_socket import ZmqSocket
from sg.scripts.realsense_recorder import RecorderImage
from sg.scripts.rgb_feature_match import RGBFeatureMatch
from sg.memory.instances.instance_process import InstanceProcess
from sg.memory.scene_graph.scene_graph_processer import SceneGraphProcesser
from transforms3d.quaternions import mat2quat

from ace.train_ace import train_ace as _train_ace
from ace.test_ace import test_ace as _test_ace
import threading
import cv2
import subprocess
import torch
from pathlib import Path
from tqdm import tqdm
import os
import shutil
import open3d as o3d
import copy
from PIL import Image
import random
import glob
import pickle
import time
import json
from typing import Union, List

def set_seed(seed=999):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
set_seed()

class Controller():
    def __init__(
            self,
            step: int=0,
            tags: str="r3d_new",
            # view dataset
            interval: int=5,
            # navigation
            min_height: float=0.1,
            resolution: float=0.01,
            occ_avoid_radius: float=0.4,  # 0.4
            # clear_around_bot_radius: float=0.2,
            occ_threshold: int=100,
            conservative: bool=True,
            # semantic memory
            box_threshold: float=0.2,
            text_threshold: float=0.2,
            nms_threshold: float=0.5,
            delete_rate: float=0.5,

            save_memory: bool=True,
            debug: bool=False,  # for debug mode, use history data
            # if delete_object_bias set to True,  this step will can't be re build
            delete_object_bias: bool=False,

            # find outlier point threshold
            nb_neighbors: int=35,
            std_ratio: float=1.5,

            socket_ip: str="192.168.1.50",
            socket_port: str="9999",
            serial: str="0"
        ):

        self.step = step
        self.interval = interval
        self.min_height = min_height
        self.max_height = min_height + 1.5
        self.resolution = resolution
        self.occ_avoid_radius = occ_avoid_radius
        self.occ_threshold = occ_threshold
        # self.clear_around_bot_radius = clear_around_bot_radius
        self.conservative = conservative
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.nms_threshold = nms_threshold
        self.serial = serial
        self.delete_rate = delete_rate
        self.save_memory = save_memory
        self.debug = debug
        self.delete_object_bias = delete_object_bias

        self.nb_neighbors = nb_neighbors
        self.std_ratio = std_ratio

        self.socket = ZmqSocket(ip=socket_ip, port=socket_port)

        self.tags = tags
        self.recorder_dir = RECORDER_DIR / self.tags

        self.suffix = f"{self.interval}_{self.min_height}_{self.resolution}_{self.conservative}_{self.box_threshold}_{self.nms_threshold}"

        self._memory_dir = self.recorder_dir / "memory" / self.suffix
        self.ace_network_path = self.recorder_dir / "ace/ace.pt"
        
        if self.step > 0 and not (self._memory_dir / f"step_{self.step}").exists():
            print(f"Error: step at {self.step} is invalid.")
            exit()

        # create memory floder
        self.create_memory_floder()

        # object memory
        self.view_dataset = None
        self.semantic_memory = None

        self.instance_objects = None
        self.instance_scene_graph = None
        self.classes_and_colors = None

        self.instance_localizer = None
        self.pathplanning = None

        self.part_level_classes = ["handle"]

        # instance object process bias indexes
        self.object_filter_indexes = None

        self.lightglue_features = None


        # self.name_to_task = {
        #     "Go to": self.go_to,
        #     "Pick up": self.pick_up,
        #     "Place": self.place
        # }

    def create_memory_floder(self):

        self.memory_dir = self._memory_dir / f"step_{self.step}"

        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        self.view_dataset_path = self.memory_dir / f"view_dataset.pkl"

        # semantic memory
        self.visualization_dir = self.memory_dir / "visualize"
        self.semantic_memory_dir = self.memory_dir / f"semantic_memory"
        self.classes_and_colors_path = self.memory_dir / "classes_and_colors.json"

        # instance memory
        self.instance_objects_path = self.memory_dir / f"instance_objects.pkl"
        self.instance_scene_graph_path = self.memory_dir / f"instance_scene_graph.pkl"

        # image retrival features
        self.lightglue_features_path = self.memory_dir / f"lightglue_features.pt"

        # see frames floder, the newest scene is save at last step last
        self.observations_dir = self.memory_dir / "observations"


    def data_collection(self):
        if self.recorder_dir.exists():
            if input("Do you want to record data? [y/n]: ") == "n":
                return 
        imagerecorder = RecorderImage(recorder_dir=self.recorder_dir, serial_number=self.serial)
        input("\033[32mPress any key to Start.\033[0m")
        record_thread = threading.Thread(target=imagerecorder.start_record)
        record_thread.start()
        input("\033[31mRecording started. Press any key to stop.\033[0m")
        imagerecorder.stop_record()
        record_thread.join()
        imagerecorder.change_size()
        imagerecorder.set_metadata()
        intrinsic = imagerecorder.intrinsic
        with open(self.recorder_dir / "calib.txt", "w") as f:
            f.write(f'{intrinsic.fx} {intrinsic.fy} {intrinsic.ppx} {intrinsic.ppy}')
        # imagerecorder.depth_to_color_vis()
        print(f"All Images are save in {imagerecorder.recorder_dir}: depth / rgb, length is {len(os.listdir(imagerecorder.recorder_dir / 'depth'))}")
        del imagerecorder
        # memroy will be delete by RecorderImage
        self.memory_dir.mkdir(parents=True, exist_ok=True)
    
    def pose_estimation(self):
        # makedirs for poses estimation
        print("\n\nPose Estimation in progress, please waiting for a moment...\n\n")
        process = subprocess.Popen([
            "conda", "run", "-n", "droidslam", "python", "pose_estimation.py",
            "--datadir", str(self.recorder_dir),
            "--calib", str(self.recorder_dir / "calib.txt"),
            "--pose_path", "poses_droidslam",
            # "--stride", str(self.interval)
            "--stride", "1"
        ], cwd="scripts", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        process.wait()

    def ransac_plane_fitting(self, points, threshold=0.04, max_iterations=1000):
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        plane_model, inliers = point_cloud.segment_plane(
            distance_threshold=threshold, ransac_n=3, num_iterations=max_iterations)

        [a, b, c, d] = plane_model
        normal_vector = np.array([a, b, c])
        return normal_vector, inliers

    def process_floor_points(self, floor_points):
        normal_vector, inliers = self.ransac_plane_fitting(floor_points)
        floor_inlier_points = floor_points[inliers]
        z_axis = np.array([0, 0, 1])
        normal_vector_normalized = normal_vector / np.linalg.norm(normal_vector)
        rotation_axis = np.cross(normal_vector_normalized, z_axis)
        rotation_angle = np.arccos(np.dot(normal_vector_normalized, z_axis))
        if np.linalg.norm(rotation_axis) > 0:
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * rotation_angle)
        else:
            rotation_matrix = np.eye(3)
        rotated_floor_inlier_points = floor_inlier_points.dot(rotation_matrix.T)
        mean_floor_z = np.mean(rotated_floor_inlier_points[:, 2])
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[2, 3] = -mean_floor_z
        return transformation_matrix

    def transform_pose_with_floor(self, display_result=False):
        # ### Initialize the GroundingDINO SAM2 model ###
        from sg.perception.models.mygroundingdinosam2 import MyGroundingDINOSAM2
        mygroundingdino_sam2 = MyGroundingDINOSAM2(
            box_threshold=0.8,
            text_threshold=0.8,
            nms_threshold=0.5
        )
        # case just has poses_droidslam, so we can't to use viewdataset,
        # we get data from floder
        mask_dir = self.recorder_dir / "mask"
        point_dir = self.recorder_dir / "point"
        poses_dir = self.recorder_dir / "poses_droidslam"
        # poses_dir = self.recorder_dir / "poses"
        rgb_dir = self.recorder_dir / "rgb"
        maks_filepaths = sorted(mask_dir.iterdir(), key=lambda x: int(x.stem))
        point_filepaths = sorted(point_dir.iterdir(), key=lambda x: int(x.stem))
        poses_filepaths = sorted(poses_dir.iterdir(), key=lambda x: int(x.stem))
        rgb_filepaths = sorted(rgb_dir.iterdir(), key=lambda x: int(x.stem))
        floor_xyzs = []
        floor_rgbs = []
        # use all iamges
        for cnt in tqdm(range(0, len(rgb_filepaths), self.interval), desc="get floor pcd and transform scene."):
            point = np.load(point_filepaths[cnt])
            image = np.asarray(Image.open(rgb_filepaths[cnt]), dtype=np.uint8)
            mask = np.load(maks_filepaths[cnt])
            pose = np.loadtxt(poses_filepaths[cnt])
            color = image / 255
            detections = mygroundingdino_sam2.run(
                image=image,
                classes=["floor"]
            )
            if len(detections.class_id) > 0:
                if display_result:
                    annotated_image, _ = mygroundingdino_sam2.vis_result(image, detections, classes=["floor"])
                    Image.fromarray(annotated_image).show()
                masks = detections.mask
                point_world = point @ pose[:3, :3].T + pose[:3, 3]
                for pred_mask in masks:
                    mask_new = np.logical_and(pred_mask, mask)
                    floor_xyzs.append(point_world[mask_new])
                    floor_rgbs.append(color[mask_new])

        floor_xyzs = np.vstack(floor_xyzs)
        floor_rgbs = np.vstack(floor_rgbs)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(floor_xyzs)
        pcd.colors = o3d.utility.Vector3dVector(floor_rgbs)
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([pcd, coordinate_frame])

        R_x_180 = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])
        tf_matrix = self.process_floor_points(floor_xyzs)

        poses_dir_new = self.recorder_dir / "poses"
        poses_dir_new.mkdir(parents=True, exist_ok=True)
        for pose_filepath in poses_filepaths:
            pose = np.loadtxt(pose_filepath)
            pose_transform = np.dot(R_x_180, np.dot(tf_matrix, pose))
            np.savetxt(poses_dir_new / pose_filepath.name, pose_transform)
        
        # empty cuda cache, we run it on lenovo Y9000K NVIDIA-RTX-4090 GPU with 16GB Memory
        del mygroundingdino_sam2
        torch.cuda.empty_cache()


    def show_pointcloud(self, is_visualize=True):
        if self.view_dataset is not None:
            pcd = self.view_dataset.index_to_pcd(list(self.view_dataset.indexes_colors_mapping_dict.keys()))
        else:
            # cache_path = self.memory_dir / f"pointcloud.ply"
            # if cache_path.exists():
            #     print(f"\n\nFound exist {cache_path}, loading it!\n\n")
            #     pcd = o3d.io.read_point_cloud(str(cache_path))
            # else:
            point_dir = self.recorder_dir / "point"
            rgb_dir = self.recorder_dir / "rgb"
            mask_dir = self.recorder_dir / "mask"
            poses_dir = self.recorder_dir / "poses"
            pcd = o3d.geometry.PointCloud()
            for index in tqdm(range(0, len(list(rgb_dir.iterdir())), self.interval), desc="point cloud"):
                img_path = rgb_dir / f"{index:06}.jpg"
                point_path = point_dir / f"{index:06}.npy"
                pose_path = poses_dir / f"{index:06}.txt"
                mask_path = mask_dir / f"{index:06}.npy"
                image = np.asarray(Image.open(img_path), dtype=np.uint8)
                point = np.load(point_path)
                pose = np.loadtxt(pose_path)
                mask = np.load(mask_path)
                rgb = image / 255
                point_world = point @ pose[:3, :3].T + pose[:3, 3]
                
                _pcd = o3d.geometry.PointCloud()
                _pcd.points = o3d.utility.Vector3dVector(point_world[mask])
                _pcd.colors = o3d.utility.Vector3dVector(rgb[mask])
                pcd += _pcd
                # cache_path.parent.mkdir(exist_ok=True)
                # o3d.io.write_point_cloud(str(cache_path), pcd)

        if is_visualize:
            pcd_downsample = pcd.voxel_down_sample(0.05)
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
            o3d.visualization.draw_geometries([pcd_downsample, coordinate_frame])

        return pcd

    def show_droidslam_pointcloud(self, use_inlier_mask=False, is_visualize=False, voxel_size=0.01):
        cache_path = self.memory_dir / f"pointcloud_droidslam_{use_inlier_mask}.ply"
        if cache_path.exists():
            print(f"\n\nFound exist {cache_path}, loading it!\n\n")
            pcd = o3d.io.read_point_cloud(str(cache_path))
        else:
            point_dir = self.recorder_dir / "point"
            rgb_dir = self.recorder_dir / "rgb"
            mask_dir = self.recorder_dir / "mask"
            poses_dir = self.recorder_dir / "poses_droidslam"
            pcd = o3d.geometry.PointCloud()
            for index in tqdm(range(0, len(list(rgb_dir.iterdir())), self.interval), desc="point cloud"):
                img_path = rgb_dir / f"{index:06}.jpg"
                point_path = point_dir / f"{index:06}.npy"
                pose_path = poses_dir / f"{index:06}.txt"
                mask_path = mask_dir / f"{index:06}.npy"
                image = np.asarray(Image.open(img_path), dtype=np.uint8)
                point = np.load(point_path)
                pose = np.loadtxt(pose_path)
                mask = np.load(mask_path)
                rgb = image / 255
                if use_inlier_mask:
                    inlier_mask = get_inlier_mask(
                        point=point, 
                        color=rgb, 
                        mask=mask, 
                        nb_neighbors=self.nb_neighbors, 
                        std_ratio=self.std_ratio
                        )
                    mask = mask * inlier_mask
                point_world = point @ pose[:3, :3].T + pose[:3, 3]
                
                _pcd = o3d.geometry.PointCloud()
                _pcd.points = o3d.utility.Vector3dVector(point_world[mask])
                _pcd.colors = o3d.utility.Vector3dVector(rgb[mask])
                pcd += _pcd
            cache_path.parent.mkdir(exist_ok=True)
            o3d.io.write_point_cloud(str(cache_path), pcd)

        if is_visualize:
            # downsampled_pcd = pcd.voxel_down_sample(voxel_size)
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
            # o3d.visualization.draw_geometries([downsampled_pcd, coordinate_frame])
            o3d.visualization.draw_geometries([pcd, coordinate_frame])
        return pcd
    

    def show_observations(self, observations, is_visualize=True):
        pcds = []
        for name, obs in observations.items():
            rgb = obs["rgb"]
            point = obs["point"]
            mask = obs["mask"]
            pose = obs["pose"]
            # if "inlier_mask" in obs.keys() and obs["inlier_mask"] is not None:
            #     mask = mask * obs["inlier_mask"]
            # Need to filter the points outside the memory range
            point_world = point @ pose[:3, :3].T + pose[:3, 3]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_world[mask])
            pcd.colors = o3d.utility.Vector3dVector(rgb[mask])
            # downsampled_pcd = pcd.uniform_down_sample(every_k_points=10)
            pcds.append(pcd)
        if is_visualize:
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
            o3d.visualization.draw_geometries([coordinate_frame] + pcds)
        return pcds

    def show_pointcloud_for_align(self, observations, is_visualize=False, voxel_size=0.03, T_matrix=None):
        original_pcd = self.show_pointcloud(is_visualize=is_visualize)
        obs_pcds = self.show_observations(observations, is_visualize=False)

        # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
        # if T_matrix is not None:
        #     coordinate_frame.transform(T_matrix)
        # o3d.visualization.draw_geometries([coordinate_frame] + obs_pcds)
        if True:
            original_pcd, ind = original_pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.5)

        # o3d.visualization.draw_geometries(obs_pcds)
        # o3d.visualization.draw_geometries([original_pcd])

        original_pcd = original_pcd.voxel_down_sample(voxel_size)
        # o3d.visualization.draw_geometries([original_pcd, coordinate_frame] + obs_pcds)
        o3d.visualization.draw_geometries([original_pcd] + obs_pcds)
        

    def train_ace(self):
        print("Train ACE")
        _train_ace(self.recorder_dir, self.ace_network_path)
        print("Train ACE Over!")
        import logging
        logging.basicConfig(level=logging.NOTSET)
    
    
    def clear_folders(self, folder_paths: list[Path]):
        for folder in folder_paths:
            if folder.is_dir():
                for item in folder.iterdir():
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()

    def get_pathplanning(self):
        self.pathplanning = PathPlanning(
            view_dataset=self.view_dataset,
            memory_dir=self.memory_dir,
            resolution=self.resolution * 5,  # A* needed bigger resolution 
            occ_avoid_radius=self.occ_avoid_radius,
            min_height=self.min_height,
            conservative=self.conservative,
            occ_threshold=self.occ_threshold
        )

    def show_start(self, point):
        point_cloud = self.show_pointcloud(is_visualize=False)
        start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
        start_sphere.translate(point)
        start_sphere.paint_uniform_color([0, 1, 0])
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([point_cloud, start_sphere, coordinate_frame])
    
    def show_target(self, A: str, B: str=None):
        target_point = self.instance_localizer.localize_AonB(A, B)
        # target_point = self.instance_localizer.localize(A)
        point_cloud = self.show_pointcloud(is_visualize=False)
        start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
        start_sphere.translate(target_point)
        start_sphere.paint_uniform_color([1, 0, 0])
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([point_cloud, start_sphere, coordinate_frame])

    def get_observations(self, just_wrist=True, save_name=None):
        if save_name is not None:
            save_path = self.observations_dir / f"{save_name}.npy"
        else:
            save_path = self.observations_dir / f"{time.time()}.npy"
        print("observation save path:", save_path)
        if self.debug and save_path.exists():
            observations = np.load(save_path, allow_pickle=True).item()
        else:
            self.socket.send_info(info={"just_wrist": just_wrist}, type="get_observations")
            observations = self.socket.received()
            self.observations_dir.mkdir(exist_ok=True)
            np.save(save_path, observations)
        
        observations_new = {}
        for i in range(len(observations["wrist"])):
            observations_new[i] = observations["wrist"][i]
        
        # top is not always useful, delete it.
        # if observations["top"] is not None:
        #     observations_new[len(observations["wrist"])] = observations["top"]

        pcds = []
        for name, obs in observations_new.items():
            # Get the rgb, point cloud, and the camera pose
            color = obs["rgb"]
            point = obs["point"]
            mask = obs["mask"]
            c2b = obs["c2b"]

            point_new = point @ c2b[:3, :3].T + c2b[:3, 3]

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_new[mask])
            pcd.colors = o3d.utility.Vector3dVector(color[mask])
            pcds.append(pcd)

            # mask_new = np.ones_like(mask, dtype=bool)
            # mask_new[300:,450:750] = False
            # image = (color*255).astype(np.uint8)
            # image[~mask_new] = [0, 0, 0]
            # Image.fromarray(image).show()

            # o3d.visualization.draw_geometries([pcd])
        coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
        # o3d.visualization.draw_geometries(pcds + [coordinate])
        o3d.visualization.draw_geometries(pcds)

        return save_path, observations_new

    def test_ace(self, observations: dict):
       estimation_poses = _test_ace(self.ace_network_path, observations)
       import logging
       logging.basicConfig(level=logging.NOTSET)
       return estimation_poses


    def self_align_observations(self, observations):
        """
        pose: pose in world coord
        T_end_in_base: pose in robot base coord
        """
        # Align the observations to the first camera if there are multiple viewpoint
        target_name = 0
        target = None
        sources = {}
        # T_cam_in_base_dict = {}
        for name, obs in observations.items():
            # Get the rgb, point cloud, and the camera pose
            color = obs["rgb"]
            point = obs["point"]
            mask = obs["mask"]
            c2b = obs["c2b"]
            
            point_base = point @ c2b[:3, :3].T + c2b[:3, 3]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_base[mask])
            pcd.colors = o3d.utility.Vector3dVector(color[mask])
            # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
            # o3d.visualization.draw_geometries([pcd, coordinate_frame])
            if name == target_name:
                target = pcd
            else:
                sources[name] = pcd
                
        blue = "\033[94m"
        green = "\033[92m"
        reset = "\033[0m"

        # remove invalid observation
        observations_new = {}
        observations_new[target_name] = copy.deepcopy(observations[target_name])
        if len(sources) > 0:
            # When there is more than one camera
            for name, source in sources.items():
                print(f"{blue}try {name} aligned with {target_name}{reset}")
                # is_sigle means just has one image, it is difficult to found correspondences when radius is too big
                is_success, transformation = self.calculate_alignment_colored_icp(source, target)
                if is_success:
                    observations_new[name] = copy.deepcopy(observations[name])
                    observations_new[name]["c2b"] = np.dot(
                        transformation, observations_new[name]["c2b"]
                    )
                    print(f"{green}Frame {name} aligned with {target_name} successfully{reset}")
                else:
                    print(f"delete invalid image {name}")
                    # Image.fromarray((observations[name]["rgb"] * 255).astype(np.uint8)).show()


        pcds = []
        for name, obs in observations_new.items():
            # Get the rgb, point cloud, and the camera pose
            color = obs["rgb"]
            point = obs["point"]
            mask = obs["mask"]
            c2b = obs["c2b"]

            point_new = point @ c2b[:3, :3].T + c2b[:3, 3]

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_new[mask])
            pcd.colors = o3d.utility.Vector3dVector(color[mask])
            pcds.append(pcd)
        coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
        o3d.visualization.draw_geometries(pcds + [coordinate])

        return observations_new

    def calculate_alignment_colored_icp(self, source, target):
        # This function is to calculate the alignment between the source and target point cloud
        # The source and target are both colored point cloud
        # The output is the transformation from source to target
        # The transformation is a 4x4 matrix
        # The function is based on the http://www.open3d.org/docs/release/tutorial/pipelines/colored_pointcloud_registration.html

        # voxel_radius = [0.04, 0.02, 0.01, 0.005, 0.0025]
        # max_iter = [50, 30, 14, 7, 3]

        # voxel_radius = [0.04, 0.02, 0.01, 0.005, 0.0025]
        # max_iter = [70, 50, 30, 10, 5]
        voxel_radius = [0.04, 0.02, 0.01]
        max_iter = [50, 30, 14]
        red = "\033[91m"
        reset = "\033[0m"
        is_success = False
        transformation = np.identity(4)
        # multi scale ICP match
        for idx in range(len(voxel_radius)):
            iter = max_iter[idx]
            radius = voxel_radius[idx]
            source_down = source.voxel_down_sample(radius)
            target_down = target.voxel_down_sample(radius)
            try:
                source_down.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 3, max_nn=30)
                )
                target_down.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 3, max_nn=30)
                )
                result_icp = o3d.pipelines.registration.registration_colored_icp(
                    source_down,
                    target_down,
                    radius,
                    transformation,
                    o3d.pipelines.registration.TransformationEstimationForColoredICP(
                        lambda_geometric=0.9999
                    ),
                    o3d.pipelines.registration.ICPConvergenceCriteria(
                        relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=iter
                    ),
                )
                print(f"IPC Number: {len(result_icp.correspondence_set)}, {len(source_down.points)}, {len(target_down.points)}")
                if len(result_icp.correspondence_set) >= (1 / 5) * min(len(source_down.points), len(target_down.points)):
                    transformation = result_icp.transformation
                    if idx == len(voxel_radius) - 1:
                        is_success = True

            except Exception as e:
                # print(f"ICP failed at scale {voxel_radius[scale]}.")
                print(f"{red}ICP failed at scale {voxel_radius[idx]}{reset}")

        return is_success, transformation

    def correct_pose_observations(self, observations):
        featurematch = RGBFeatureMatch()
        target_index_dict = {}
        best_matches_len_list = []
        pose_inlier_num_list = []
        for name, obs in observations.items():
            rgb = obs["rgb"]
            image = (rgb * 255).astype(np.uint8)
            best_index, best_matches_len = featurematch.find_most_similar_image(
                image, features=self.lightglue_features, visualize=False, view_dataset=self.view_dataset)
            
            target_index_dict[name] = best_index
            best_matches_len_list.append(best_matches_len)
            pose_inlier_num_list.append(obs["pose_inlier_num"])

        best_matches_index = np.argmax(np.array(best_matches_len_list) * 0.1 + np.array(pose_inlier_num_list) * 0.9)

        
        # find the biggest matches ace pose for other pose refinement
        best_matches_index = np.argmax(best_matches_len_list)
        obs_keys = list(observations.keys())
        best_c2w = observations[obs_keys[best_matches_index]]["pose"]  # c2w
        best_c2b = observations[obs_keys[best_matches_index]]["c2b"]  # c2b
        b2w = best_c2w @ np.linalg.inv(best_c2b)

        # use biggest mathes ace pose and c2b to calculate other frames' pose
        for name, obs in observations.items():
            if name != best_matches_index:
                obs["pose"] = b2w @ obs["c2b"]

        source_pcds = o3d.geometry.PointCloud()
        target_pcds = o3d.geometry.PointCloud()

        for name, obs in observations.items():
            rgb = obs["rgb"]
            image = (rgb * 255).astype(np.uint8)
            point = obs["point"]
            mask = obs["mask"]
            pose = obs["pose"]
            point_w = point @ pose[:3, :3].T + pose[:3, 3]
            source_pcd = o3d.geometry.PointCloud()
            source_pcd.points = o3d.utility.Vector3dVector(point_w[mask])
            source_pcd.colors = o3d.utility.Vector3dVector(rgb[mask])
            source_pcds += source_pcd

            target_index = target_index_dict[name]
            target_rgb = (self.view_dataset.images[target_index] / 255).astype(np.float32)
            target_point_w = self.view_dataset.global_points[target_index]
            target_mask = self.view_dataset.masks[target_index]
            target_pcd = o3d.geometry.PointCloud()
            target_pcd.points = o3d.utility.Vector3dVector(target_point_w[target_mask])
            target_pcd.colors = o3d.utility.Vector3dVector(target_rgb[target_mask])
            # target_pcd.points = o3d.utility.Vector3dVector(target_point_w.reshape(-1, 3))
            # target_pcd.colors = o3d.utility.Vector3dVector(target_rgb.reshape(-1, 3))
            target_pcds += target_pcd

        if False:
            source_pcds, _ = source_pcds.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.1)
            target_pcds, _ = target_pcds.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.1)
            
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
            o3d.visualization.draw_geometries([target_pcds, coordinate_frame])

        is_success, ref_tf_matrix = self.calculate_alignment_colored_icp(source_pcds, target_pcds)
        if is_success:
            for name, obs in observations.items():
                obs["pose"] = np.dot(ref_tf_matrix, obs["pose"])

        del featurematch
        return observations, is_success


    def get_align_observations(self, use_inlier_mask=False, show_align=False, just_wrist=False, 
                               self_align=True, align_to_world=False, save_name=None):
        print("===> get observations from robot.")

        _, observations = self.get_observations(just_wrist=just_wrist, save_name=save_name)
        
        for name, obs in observations.items():
            point = obs["point"]
            rgb = obs["rgb"]
            mask = obs["mask"]
            if use_inlier_mask:
                inlier_mask = get_inlier_mask(point=point, color=rgb, mask=mask)
                mask = np.logical_and(mask, inlier_mask)
            obs["mask"] = mask
            # obs["pose"] = rough_poses[name]

        if self_align:
            observations = self.self_align_observations(observations)
            
        is_success = True
        # align observations from base coord to world coord
        if align_to_world:
            rough_poses = self.test_ace(observations)
            for index, (name, obs) in enumerate(observations.items()):
                obs["pose"], obs["pose_inlier_num"] = rough_poses[index]
            observations, is_success = self.correct_pose_observations(observations)
            if show_align and is_success:
                self.show_pointcloud_for_align(observations)
        return observations, is_success

    def localize_robot(self, observations, show=True):
        T_base_in_world = self.get_T_base_in_world(
            observations[0]["pose"], 
            observations[0]["T_end_in_base"]
        )
        if show:
            self.show_start(T_base_in_world[:3, 3])
        # start_xy = T_base_in_world[:2, 3].tolist()
        return T_base_in_world
    
    def generation_paths(self, robot_pose: np.ndarray, tar_object: str, ref_objec: str=None):
        start = robot_pose[:2, 3]
        _paths = self.pathplanning.generate_paths(start, A=tar_object, B=ref_objec)
        assert robot_pose.shape == (4, 4)
        assert _paths is not None  # TODO
        paths = self.process_paths(_paths, robot_pose)
        return paths
    
    def pose_to_euler(self, pose: np.ndarray):
        from scipy.spatial.transform import Rotation as R
        euler_angles_zyx = R.from_matrix(pose[:3, :3]).as_euler('ZYX')
        pose_dict = {
            'x': str(pose[0, 3]),
            'y': str(pose[1, 3]),
            'z': str(pose[2, 3]),
            'rx': str(euler_angles_zyx[2]),
            'ry': str(euler_angles_zyx[1]),
            'rz': str(euler_angles_zyx[0])
        }
        return pose_dict
    

    def project_point_to_pixel(self, points, intrinsic):
        # The points here should be in the camera coordinate, n*3
        points = np.array(points)
        pixels = []

        pixels = cv2.projectPoints(
            points,
            np.zeros(3),
            np.zeros(3),
            intrinsic,
            np.zeros(5)
        )[0][:, 0, :]

        return pixels[:, ::-1]


    def get_instance_localizer(self):
        # init when need pathplanning
        self.instance_localizer = InstanceLocalizer(
            view_dataset=self.view_dataset,
            instances_objects=self.instance_objects,
            device="cuda"
        )

    def get_view_dataset(self):
        # use_inlier_mask will slow process speed but will get well pcd
        
        if self.view_dataset_path.exists():
            print(self.view_dataset_path)
            print("\n\nFound cache view_dataset, loading it!\n\n")
            with open(self.view_dataset_path, 'rb') as f:
                self.view_dataset = pickle.load(f)
        else:
            from sg.memory.view_dataset import ViewDataset
            self.view_dataset = ViewDataset(
                self.recorder_dir, 
                interval=self.interval, 
                resolution=self.resolution,
                nb_neighbors=self.nb_neighbors,
                std_ratio=self.std_ratio
            )
            # save at step 0 to avoid a bug that requires you to start over
            if True and self.step == 0:
                # save view dataset
                with open(self.view_dataset_path, 'wb') as f:
                    pickle.dump(self.view_dataset, f, protocol=4)

    def get_semantic_memory(
            self,
            device: float="cuda",
            visualize_results: bool=True,
    ):
        ## in this function, classes_and_colors also been getted and save
        # this function is only memory, save when after process
        if self.semantic_memory_dir.exists() and \
                len(list(self.semantic_memory_dir.iterdir())) == self.view_dataset.append_length_log[-1]:
                print("\n\nFound cache semantic_memory, don't need process!\n\n")
                with open(self.classes_and_colors_path, "r") as f:
                    self.classes_and_colors = json.load(f)
                return

        if visualize_results:
            self.visualization_dir.mkdir(parents=True, exist_ok=True)
        self.semantic_memory_dir.mkdir(exist_ok=True)
        from sg.memory.ram_groundingdino_sam2_clip_semantic_memory import RamGroundingDinoSAM2ClipDataset

        # based on append log for image retrival
        append_length = self.view_dataset.append_length_log[-1]
        images = self.view_dataset.images[-append_length:]
        names = self.view_dataset.names[-append_length:]

        if self.classes_and_colors is None:
            self.classes_and_colors = {
                "classes": [],
                "class_colors": {}
            }

        semantic_memory = RamGroundingDinoSAM2ClipDataset(
            classes=self.classes_and_colors["classes"],
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            nms_threshold=self.nms_threshold,
            device=device,
        )

        with torch.no_grad():
            for cnt in tqdm(range(len(images)), total=len(images), desc="semantic meomry"):
                image = images[cnt]
                name = names[cnt]
                det_res, annotated_image, image_pil = semantic_memory.semantic_process(image=image)
                
                if visualize_results:
                    assert self.visualization_dir is not None
                    cv2.imwrite(str(self.visualization_dir / f"{name}.jpg"), 
                    cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
                    image_pil.save(self.visualization_dir / f"{name}_Clean.jpg")
                detection_save_path = self.semantic_memory_dir / f"{name}.pkl"

                with open(detection_save_path, "wb") as f:
                    pickle.dump(det_res, f)

        self.classes_and_colors = semantic_memory.get_classes_and_colors()
        with open(self.classes_and_colors_path, "w") as f:
            json.dump(self.classes_and_colors, f)

        # clear gpu memory
        del semantic_memory
        torch.cuda.empty_cache()

    def get_instances(self): 
         # updata view dataset by objects delete indexes if update_view is True
        if self.instance_objects_path.exists():
            with open(self.instance_objects_path, "rb") as f:
                self.instance_objects = pickle.load(f)
            print(f"Exsit instances objects in {self.instance_objects_path}")
        else:
            instance_process = InstanceProcess(
                downsample_voxel_size=self.resolution,
                part_level_classes=self.part_level_classes
            )
            self.instance_objects, self.object_filter_indexes = instance_process.get_instances(
                memory_dir=self.memory_dir,
                view_dataset=self.view_dataset,
                # classes=self.classes_and_colors["classes"],
            )

            del instance_process

            # save at step 0 to avoid a bug that requires you to start over
            if True and self.step == 0:
                # save instance objects
                with open(self.instance_objects_path, "wb") as f:
                    pickle.dump(self.instance_objects, f)
        

    def get_instance_scene_graph(self, is_visualize=True):
        if self.instance_scene_graph_path.exists():
            with open(self.instance_scene_graph_path, "rb") as f:
                self.instance_scene_graph = pickle.load(f)
        else:
            scenegraphprocesser = SceneGraphProcesser(
                part_level_classes=self.part_level_classes,
                resolution=self.resolution
            )
            self.instance_scene_graph = scenegraphprocesser.build_scene_graph(
                view_dataset=self.view_dataset,
                instance_objects=self.instance_objects 
            )

            # save at step 0 to avoid a bug that requires you to start over
            if True and self.step == 0:
                # save instance scene graph
                with open(self.instance_scene_graph_path, "wb") as f:
                    pickle.dump(self.instance_scene_graph, f)

        self.instance_scene_graph.visualize(save_dir=self.memory_dir)

    def get_lightglue_features(self):
        if self.lightglue_features_path.exists():
            self.lightglue_features = torch.load(self.lightglue_features_path) 
        else:
            featurematch = RGBFeatureMatch()
            append_length = self.view_dataset.append_length_log[-1]
            images = self.view_dataset.images[-append_length:]
            self.lightglue_features = featurematch.extract_memory_features(images=images, features=self.lightglue_features)
            torch.save(self.lightglue_features, self.lightglue_features_path)

 
    def get_task_plan(self, description: str, change_level: str):
        # each task will create new floder to save memory
        # so if you had over a long-term task, you should edit scene to step 0
        self._memory_dir = self._memory_dir / f"{change_level} long_term_task: {description}"
        self._memory_dir.mkdir(exist_ok=True)
        self.create_memory_floder()

        taskplanning = TaskPlanning(save_dir=self._memory_dir)
        response = taskplanning.get_response(description=description)
        tasks = response["subtasks"]
        return tasks

    def get_current_position(self, observations: dict):
        obs0 = observations[0]
        c2w0 = obs0["pose"]
        c2b0 = obs0["c2b"]
        b2w = c2w0 @ np.linalg.inv(c2b0)     

        ############ base in agv transform ################
        b2agv = np.array([
            [1, 0, 0, 0.183],
            [0, 1, 0, 0],
            [0, 0, 1, 0.1725],
            [0, 0, 0, 1]
        ])
        ############ base in agv transform ################
        agv2w = b2w @ np.linalg.inv(b2agv)

        translation = agv2w[:3, 3]
        rotation = agv2w[:3, :3]
        
        if True:
            self.show_start(translation)

        return translation, rotation

    def go_to(
            self, object1: str, 
            object2: str, 
            start_point: np.ndarray, 
            start_rotation: np.ndarray,
            target_point=None
        ):
        print(f"Runing Go to({object1}, {object2}) Task.")

        # start_point, start_rotation = self.get_current_position(observations)
        initial_theta = np.arctan2(start_rotation[1, 0], start_rotation[0, 0])
        # self.show_start(start_point)
        # self.show_target(object1, object2)
        if target_point is None:
            target_point = self.instance_localizer.localize_AonB(object1, object2)
        paths_ = self.pathplanning.generate_paths(start_point, target_point, visualize=True)
        paths = np.array(paths_)

        first_angle = paths[0, 2] - initial_theta
        paths[:3, 2] -= paths[0, 2]
        paths[:3, 2] += first_angle

        if self.debug or True:
            input("please move the agent to target point (Press Enter).")
        else:
            self.socket.send_info(info=paths, type="motion_moving")
            response = self.socket.received()
            print(response)
        

    def pick_up(self, object1: str, observations: dict, is_visualize=False):
        print(f"Runing Pick up({object1}) Task.")
        # on each pick up or place, re-set object_handler to spend for GPU memory save
        from sg.manipulation.objecthandler import ObjectHandler
        object_handler = ObjectHandler(
            box_threshold=0.6,
            text_threshold=0.6,
            device="cuda",
            is_visualize=is_visualize
        )

        target_pose = object_handler.pickup(
            observations=observations,
            query=object1
        )

        if self.debug:
            input("please make sure object is been pickup (Press Enter).")
        else:
            self.socket.send_info(info={
                "action_parameters": target_pose,
                "speed": 100,
                }, type="pick_up")
            response = self.socket.received()
            print(response)
            if "Exception response" in response.keys():
                print("robot arm exception, this task error!")
                exit(0)   

        del object_handler
        torch.cuda.empty_cache()

    def place(self, object1: str=None, object2: str=None, observations: dict={}, is_visualize=False):
        print(f"Runing Place({object1}, {object2}) Task.")
        # on each pick up or place, re-set object_handler to spend for GPU memory save
        from sg.manipulation.objecthandler import ObjectHandler
        object_handler = ObjectHandler(
            box_threshold=0.2,
            text_threshold=0.2,
            device="cuda",
            is_visualize=is_visualize
        )

        place_pose = object_handler.place(
            observations=observations,
            object1=object1,
            object2=object2
        )

        # get back container object
        save_path = self.observations_dir / f"pick_back_{object1}.npy"

        if self.debug and save_path.exists():
            back_observation = np.load(save_path, allow_pickle=True).item()
        else:
            self.socket.send_info(info="", type="get_back_observation")
            back_observation = self.socket.received()
            self.observations_dir.mkdir(exist_ok=True)
            np.save(save_path, back_observation)

        pick_back_objct_pose = object_handler.pick_back_object(back_observation=back_observation, query=object1)


        # pick back object
        if self.debug:
            input("please make sure back object is been pick up (Press Enter).")
        else:
            self.socket.send_info(info={
                "action_parameters": pick_back_objct_pose,
                "speed": 100,
                "pick_back": True
                }, type="pick_up")
            response = self.socket.received()
            print(response)

        # place object
        if self.debug:
            input("please make sure object is been place.")
        else:
            self.socket.send_info(info={
                "action_parameters": place_pose,
                "speed": 100,
                }, type="place")
            response = self.socket.received()
            print(response)


        del object_handler
        torch.cuda.empty_cache()

    def run_tasks(self, tasks: Union[List[dict]]):
        self.get_instance_localizer()
        self.get_pathplanning()
        observations, correct_success = self.get_align_observations(
            just_wrist=True,
            show_align=True,
            use_inlier_mask=True,
            self_align=False,
            align_to_world=True,
            save_name="0_start",
        )
        if not correct_success:
            assert 1 == 0, "Init Pose Error!"

        init_position, init_rotation = self.get_current_position(observations)
        current_position = init_position
        current_rotation = init_rotation
        for index, task in enumerate(tasks):
            print(f"\n\nNow are in step {self.step}\n\n")
            start_time = time.time()
            if task["action"] == "Go to":
                if task["object1"] == "back":
                    # update instance localizer and pathplanning
                    self.go_to(object1=task["object1"], object2=task["object2"], \
                               start_point=current_position, start_rotation=current_rotation, target_point=init_position)
                else:
                    self.go_to(object1=task["object1"], object2=task["object2"], \
                               start_point=current_position, start_rotation=current_rotation)
                save_name = f"{index + 1}_after_{task['action']}({task['object1']}, {task['object2']})"
            elif task["action"] == "Pick up":
                self.pick_up(object1=task["object1"], observations=observations, is_visualize=True)
                save_name = f"{index + 1}_after_{task['action']}({task['object1']})"
            elif task["action"] == "Place":
                self.place(object1=task["object1"], object2=task["object2"], observations=observations, is_visualize=True)
                save_name = f"{index + 1}_after_{task['action']}({task['object1']}, {task['object2']})"
            else:
                raise NotImplementedError
            
            # Just use wrist camera, which is enough
            just_wrist = True
            observations, correct_success = self.get_align_observations(
                just_wrist=just_wrist,
                show_align=True,
                use_inlier_mask=True,
                self_align=False,
                align_to_world=True,
                save_name=save_name
            )
            current_position, current_rotation = self.get_current_position(observations)
            # just after pick up and place, update scene
            if task["action"] in ["Pick up", "Place"]:
                if not correct_success:
                    assert 1 == 0, "relocalize error!"
                    
                self.update_scene(observations=observations)
                end_time = time.time()
                print(f"Step spend time is: {end_time - start_time}")

                self.show_instances(
                    self.instance_objects, 
                    clip_vis=True, 
                    scene_graph=self.instance_scene_graph, 
                    show_background=True
                )
                if index != len(tasks) - 1:
                    self.get_instance_localizer()
                    self.get_pathplanning()
                    
        print("Long-term task execution success!")

    def show_instances(self, instance_objects, show_background=False, scene_graph=None, clip_vis=False):
        """
        instances:
            objects: is numpy object with not open3d object
            class_names
            class_colors
        """
        from sg.memory.instances.visualize_instances import vis_instances
        pcds = vis_instances(
            instance_objects=instance_objects,
            class_colors=self.classes_and_colors["class_colors"],
            view_dataset=self.view_dataset,
            instance_scene_graph=scene_graph,
            show_background=show_background,
            clip_vis=clip_vis
        )

        # if True:
        #     pcd_all = o3d.geometry.PointCloud()
        #     for pcd in pcds:
        #         pcd_all += pcd

        #     o3d.io.write_point_cloud(str(self.memory_dir / "pointcloud.ply"), pcd_all)

    def find_need_to_delete_indexes(self, observations: dict):
        depth_thres = self.resolution * 2
        color_depth_thres = self.resolution
        color_thres = 0.1

        voxel_indexes_memroy = np.array(list(self.view_dataset.indexes_colors_mapping_dict.keys()))
        voxel_points_memory = self.view_dataset.index_to_point(voxel_indexes_memroy)
        voxel_colors_memory = np.array(list(self.view_dataset.indexes_colors_mapping_dict.values()))

        """find need delete indexes"""
        need_delete_indexes = []
        for name, obs in observations.items():
            pose = obs["pose"]
            color = obs["rgb"]
            mask = obs["mask"]
            intrinsic = obs["intrinsic"]
            height, width = color.shape[:2]
            pose_inv = np.linalg.inv(pose)
            points_camera = voxel_points_memory @ pose_inv[:3, :3].T + pose_inv[:3, 3]
            # Only consider the voxels in front of the camera (OpenCV camera coordinate)
            mask = points_camera[:, 2] > 0
            pc_xy = np.zeros((points_camera.shape[0], 2), dtype=np.int64)  # int object
            pc_xy[mask] = self.project_point_to_pixel(
                points_camera[mask], intrinsic
            )
            # Only consider the voxels in the camera view
            mask *= (
                (pc_xy[:, 0] >= 0)
                * (pc_xy[:, 0] <= height - 1)
                * (pc_xy[:, 1] >= 0)
                * (pc_xy[:, 1] <= width - 1)
            )
            # depth = obs["depth"] / 1000
            depth = obs["depth"]
            # depth_valid = np.logical_and(depth > 0.35, depth < 1.2)
            # depth_mask = np.logical_and(obs["mask"], depth_valid)
            depth_mask = obs["mask"]

            # Consider the voxels whose depth is bigger than the current observation (more close to the camera)
            # The OpenCV camera coordinate, distance is positive
            min_depth_map = np.full((height, width), np.inf)
            np.minimum.at(min_depth_map, (pc_xy[mask, 0], pc_xy[mask, 1]), points_camera[mask, 2])

            depth_differ = (depth - min_depth_map)[pc_xy[mask, 0], pc_xy[mask, 1]]
            # depth_differ = (
            #     depth[pc_xy[mask, 0], pc_xy[mask, 1]] - points_camera[mask, 2]
            # )
            color_differ = ((voxel_colors_memory[mask] - obs["rgb"][pc_xy[mask, 0], pc_xy[mask, 1]])** 2).sum(1) ** 0.5
            delete_valid_mask = np.logical_or(
                depth_differ > depth_thres,
                (depth_differ > color_depth_thres) * (color_differ > color_thres),
            )
            valid_depth_mask = depth_mask[pc_xy[mask, 0], pc_xy[mask, 1]]
            delete_valid_mask[~valid_depth_mask] = False  # invalid depth can't give reference
            need_delete_indexes += voxel_indexes_memroy[mask][delete_valid_mask].tolist()

        need_delete_indexes = list(set(need_delete_indexes))
        return need_delete_indexes

    def update_view_dataset(self, observations: dict, need_delete_indexes: list):
        # update the view dataset based on need_delete_indexes and save it to the current step folder for subsequent recovery
        # ​​update the information in the list of observations in the view dataset,
        # and obtain the mapping of indexes and colors after voxelization of the information obtained from the new perspective 
        """updata view dataset"""
        self.view_dataset.length += len(observations)
        self.view_dataset.append_length_log.append(len(observations))
        new_add_indexes_colors_mapping_dict = {}
        for name, obs in observations.items():
            point = obs["point"]
            color = obs["rgb"]
            mask = obs["mask"]
            pose = obs["pose"]
            global_point = point @ pose[:3, :3].T + pose[:3, 3]

            self.view_dataset.images.append((obs["rgb"] * 255).astype(np.uint8))
            self.view_dataset.masks.append(mask)
            self.view_dataset.names.append(f"{int(self.view_dataset.names[-1]) + 1:06}")
            self.view_dataset.global_points.append(global_point)
            
            # # self.bounds now are not support change,
            # # cause it will spend lot time
            pixel_index_mapping, pixel_index_mask, agv_color, unique_indexes = self.view_dataset.voxelize(global_point, color, mask)
            self.view_dataset.pixel_index_mappings.append(pixel_index_mapping)
            self.view_dataset.pixel_index_masks.append(pixel_index_mask)

            new_add_indexes_colors_mapping_dict.update(dict(zip(unique_indexes, agv_color)))

        # udpate indexes_colors_mapping_dict based on delete indexes
        # old_indexes_colors_mapping_dict = copy.deepcopy(self.view_dataset.indexes_colors_mapping_dict)
        # remaining_indexes = list(set(self.view_dataset.indexes_colors_mapping_dict.keys()) - set(need_delete_indexes))
        remaining_indexes = np.setdiff1d(list(self.view_dataset.indexes_colors_mapping_dict.keys()), need_delete_indexes)
        remaining_colors = [self.view_dataset.indexes_colors_mapping_dict[idx] for idx in remaining_indexes]

        # based on new_add_indexes_colors_mapping_dict, update the modified indexes_colors_mapping_dict
        self.view_dataset.indexes_colors_mapping_dict = dict(zip(remaining_indexes, remaining_colors))
        self.view_dataset.indexes_colors_mapping_dict.update(new_add_indexes_colors_mapping_dict)


    
    def update_instance_objects(self, need_delete_indexes: list):
        # _new_instances = self.del_instances_by_indexes(self.instance_objects, need_delete_indexes)
        delete_objects = []
        for ins_obj in self.instance_objects:
            indexes = ins_obj["indexes"]
            inter_indexes = np.intersect1d(indexes, need_delete_indexes)
            inter_rate = len(inter_indexes) / len(indexes)

            # get object name
            values, counts = np.unique(ins_obj["class_name"], return_counts=True)
            class_name = values[np.argmax(counts)]
            # print(class_name, inter_rate)

            if inter_rate > self.delete_rate:
                delete_objects.append(ins_obj)
            else:
                # ins_obj["indexes"] = [idx for idx in indexes if idx not in inter_indexes]
                # make sure the type of indexes is list
                ins_obj["indexes"] = list(np.setdiff1d(indexes, inter_indexes))
            
        for ins_obj in delete_objects:
            self.instance_objects.remove(ins_obj)

        # Based on the new view_dataset, update semantic memory 
        # and save it in the semantic memory folder under the current setp
        self.get_semantic_memory()

        # Update instances based on the newly saved semantic memory and view_dataset
        instance_process = InstanceProcess(
            downsample_voxel_size=self.resolution,
            part_level_classes=self.part_level_classes
        )

        self.instance_objects, self.object_filter_indexes = instance_process.get_instances(
            memory_dir=self.memory_dir,
            view_dataset=self.view_dataset,
            # classes=self.classes_and_colors["classes"],
            objects=self.instance_objects,
            obj_min_detections=2,
        )

    def update_scene_graph(self):
        scenegraphprocesser = SceneGraphProcesser(
            part_level_classes=self.part_level_classes,
            resolution=self.resolution
        )

        self.instance_scene_graph = scenegraphprocesser.update_scene_graph(
            view_dataset=self.view_dataset,
            instance_objects=self.instance_objects,
            history_scene_graph=self.instance_scene_graph
        )

        self.instance_scene_graph.visualize(save_dir=self.memory_dir)

    def save_step_memroy(self):
        print(f"Saving step {self.step} memory...")
        # assert for result truth
        assert len(self.lightglue_features) == sum(self.view_dataset.append_length_log)

        if self.delete_object_bias:
            remaining_indexes = np.setdiff1d(list(self.view_dataset.indexes_colors_mapping_dict.keys()), self.object_filter_indexes)
            remaining_colors = [self.view_dataset.indexes_colors_mapping_dict[idx] for idx in remaining_indexes]
            # update the indexes_colors_mapping_dict based on object_filter_indexes
            self.view_dataset.indexes_colors_mapping_dict = dict(zip(remaining_indexes, remaining_colors))

        # save view dataset
        with open(self.view_dataset_path, 'wb') as f:
            pickle.dump(self.view_dataset, f, protocol=4)

        # save instance objects
        with open(self.instance_objects_path, "wb") as f:
            pickle.dump(self.instance_objects, f)
        
        # save instance scene graph
        with open(self.instance_scene_graph_path, "wb") as f:
            pickle.dump(self.instance_scene_graph, f)

        # save lightglue features
        torch.save(self.lightglue_features, self.lightglue_features_path)
        

    def update_scene(
        self,
        observations: dict,
        # delete_object_bias: bool=True
    ):        
        # create new step memory floder
        self.step += 1
        self.create_memory_floder()
        
        # find the indexes that need to be deleted (need delete indexes)
        print("====> find need delete indexes")
        need_delete_indexes = self.find_need_to_delete_indexes(observations=observations)
        
        # update view_dataset based on need_delete_indexes
        print("====> update view dataset")
        self.update_view_dataset(
            observations=observations, 
            need_delete_indexes=need_delete_indexes
        )

        # update lightglue features
        print("====> update lightglue_features")
        self.get_lightglue_features()

        # update instance_objects based on need_delete_indexes
        print("====> update instance objects")
        self.update_instance_objects(
            need_delete_indexes=need_delete_indexes
        )

        # update scene graph base on history scene graph and instance objects which just after update 
        print("====> update instance scene graph")
        self.update_scene_graph()

        if self.save_memory:
            self.save_step_memroy()

if __name__ == "__main__":
    c_script = Controller()
    c_script.times = 1
    c_script.show_pointcloud_relocalize()