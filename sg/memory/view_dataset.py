from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
import open3d as o3d
from dataclasses import dataclass
from sg.utils.utils import get_inlier_mask
import cv2


@dataclass(frozen=True)
class Bounds:
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float
    @property
    def xdiff(self) -> float:
        return self.xmax - self.xmin
    @property
    def ydiff(self) -> float:
        return self.ymax - self.ymin
    @property
    def zdiff(self) -> float:
        return self.zmax - self.zmin
    
    @property
    def lower_bound(self) -> np.ndarray:
        return np.array([self.xmin, self.ymin, self.zmin])

    @property
    def higher_bound(self) -> np.ndarray:
        return np.array([self.xmax, self.ymax, self.zmax])

    @classmethod
    def from_arr(cls, bounds) -> "Bounds":
        assert bounds.shape == (3, 2), f"Invalid bounds shape: {bounds.shape}"
        return Bounds(
            xmin=bounds[0, 0].item(),
            xmax=bounds[0, 1].item(),
            ymin=bounds[1, 0].item(),
            ymax=bounds[1, 1].item(),
            zmin=bounds[2, 0].item(),
            zmax=bounds[2, 1].item(),
        )

class ViewDataset():
    def __init__(
        self, 
        recorder_dir: str, 
        interval: int=1,
        use_inlier_mask: bool=True,
        resolution: float=0.01,
        nb_neighbors: int=30,
        std_ratio: float=1.5
    ):
        """For original dataset"""
        self.recorder_dir = Path(recorder_dir)
        self.interval = interval
        self.use_inlier_mask = use_inlier_mask
        self.resolution = resolution
        self.nb_neighbors = nb_neighbors
        self.std_ratio = std_ratio

        self.metadata = self.read_metadata()

        self.images = []
        self.masks = []
        self.names = []
        self.global_points = []
        
        self.bounds = None  # once been setup, can't be change

        # log append memory
        self.append_length_log = []

        self.load_data()

        self.voxel_num = ((self.bounds.higher_bound - self.bounds.lower_bound) / self.resolution).astype(np.int32)

        """For Voxel and Update"""
        self.pixel_index_mappings = []
        self.pixel_index_masks = []

        ### indexes_colors_mapping_dict and background Always include the latest scenes
        self.indexes_colors_mapping_dict = {}

        self.calculate_all_global_voxel_indexes_and_colors()

    def read_metadata(self):
        with open(self.recorder_dir / "metadata.json", "r") as f:
            metadata_dict = json.load(f)
        self.rgb_width = metadata_dict["w"]
        self.rgb_height = metadata_dict["h"]
        self.intrinsic_matrix = np.array(metadata_dict["K"]).reshape(3, 3)
        self.dist_coef = np.array(metadata_dict["dist_coef"])
        self.image_size = (self.rgb_height, self.rgb_width)
        self.length = metadata_dict["length"]

    def load_image(self, filepath):
        image = np.asarray(Image.open(self.recorder_dir / filepath), dtype=np.uint8)
        return image
    
    def load_point(self, filepath):
        return np.load(self.recorder_dir / filepath, allow_pickle=True).astype(np.float32)

    def load_mask(self, filepath):
        return np.load(self.recorder_dir / filepath, allow_pickle=True).astype(np.bool_)
    
    def load_pose(self, filepath):
        return np.loadtxt(self.recorder_dir / filepath).astype(np.float32)
    
    def load_depth(self, filepath):
        # np.int16 is needed
        # depth = np.asarray(Image.open(self.recorder_dir / filepath), dtype=np.int16)
        depth = np.load(self.recorder_dir / filepath)
        return depth

    def load_data(self):
        min_bounds = np.array([np.inf, np.inf, np.inf])
        max_bounds = np.array([-np.inf, -np.inf, -np.inf])
        for cnt in tqdm(range(0, self.length, self.interval), desc="Loading data: "):
            image_filepath = f"rgb/{cnt:06}.jpg"
            mask_filepath = f"mask/{cnt:06}.npy"
            point_filepath = f"point/{cnt:06}.npy"
            pose_filepath = f"poses/{cnt:06}.txt"

            image = self.load_image(image_filepath)
            mask = self.load_mask(mask_filepath)
            point = self.load_point(point_filepath)
            pose = self.load_pose(pose_filepath)
            
            self.images.append(image)
            if self.use_inlier_mask:
                color = image / 255
                inlier_mask = get_inlier_mask(
                    point=point, 
                    color=color, 
                    mask=mask,
                    nb_neighbors=self.nb_neighbors,
                    std_ratio=self.std_ratio
                )
                mask = np.logical_and(mask, inlier_mask)

            gpoint = point @ pose[:3, :3].T + pose[:3, 3]
            
            # caculate bounds
            points_world = gpoint[mask]
            if len(points_world) > 0:
                min_bounds = np.minimum(min_bounds, np.amin(points_world, axis=0))
                max_bounds = np.maximum(max_bounds, np.amax(points_world, axis=0))

            self.masks.append(mask)
            self.names.append(f"{cnt:06}")
            self.global_points.append(gpoint)


        xmin, ymin, zmin = min_bounds
        xmax, ymax, zmax = max_bounds
        _bounds_arr = np.array([
            [xmin, xmax],
            [ymin, ymax],
            [zmin, zmax]
        ])
        # round for two decimal place to one cm
        bounds_arr = np.round(_bounds_arr, 2)
        bounds = Bounds.from_arr(bounds_arr)

        self.bounds = bounds
        self.append_length_log.append(len(self.global_points))


    def voxelize(self, point, color, mask):
        range_mask = ((point - self.bounds.lower_bound) >= 0).all(-1) * \
                    ((self.bounds.higher_bound - point) >= 0).all(-1)
        
        pixel_index_mask = np.logical_and(mask, range_mask)

        pixel_index_mapping = -np.ones((point.shape[:2]), dtype=np.int32)
        pixel_index_mapping[pixel_index_mask] = self.point_to_index(point[pixel_index_mask])

        valid_voxel_indexes = pixel_index_mapping[pixel_index_mask]
        valid_color = color[pixel_index_mask]

        if np.any(valid_voxel_indexes == -1):
            print("Warning: the voxel index is -1. The pixel_index_mask is wrong")
        
        unique_voxel_indexes, inverse_indices, counts = np.unique(
            valid_voxel_indexes, return_inverse=True, return_counts=True
        )

        # Sum colors for each unique voxel index
        summed_color = np.zeros((unique_voxel_indexes.size, valid_color.shape[1]))
        np.add.at(summed_color, inverse_indices, valid_color)

        agv_color = summed_color / counts[:, np.newaxis]
        unique_indexes = unique_voxel_indexes

        return pixel_index_mapping, pixel_index_mask, agv_color, unique_indexes

    def calculate_all_global_voxel_indexes_and_colors(self):
        _scene_colors = []
        _scene_indexes = []
        for cnt in tqdm(range(len(self.global_points)), desc="voxel map"):
            gpoint = self.global_points[cnt]
            color = (self.images[cnt] / 255).astype(np.float32)
            mask = self.masks[cnt]

            pixel_index_mapping, pixel_index_mask, agv_color, unique_indexes = self.voxelize(point=gpoint, color=color, mask=mask)

            _scene_colors.append(agv_color)
            _scene_indexes.append(unique_indexes)

            self.pixel_index_mappings.append(pixel_index_mapping)
            self.pixel_index_masks.append(pixel_index_mask)


        print("===> get unique global points and colors")
        # the new unique to unique global points and colors
        _scene_colors = np.vstack(_scene_colors)
        _scene_indexes = np.hstack(_scene_indexes)

        unique_scene_indexes, inverse_scene_indices, scene_counts = np.unique(
            _scene_indexes, return_inverse=True, return_counts=True
        )
        summed_scene_color = np.zeros((unique_scene_indexes.size, _scene_colors.shape[1]))
        np.add.at(summed_scene_color, inverse_scene_indices, _scene_colors)
        
        voxel_scene_colors = summed_scene_color / scene_counts[:, np.newaxis]
        # voxel_scene_points = self.index_to_point(unique_scene_indexes)
        voxel_scene_indexes = unique_scene_indexes
        self.indexes_colors_mapping_dict = dict(zip(voxel_scene_indexes, voxel_scene_colors))

        if False:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.index_to_point(list(self.indexes_colors_mapping_dict.keys())))
            pcd.colors = o3d.utility.Vector3dVector(np.array(list(self.indexes_colors_mapping_dict.values())))
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
            o3d.visualization.draw_geometries([pcd, coordinate_frame])

    def point_to_voxel(self, points):
        if type(points) == list:
            points = np.array(points)
        # The points is in numpy array with shape (..., 3)
        # The voxels is in numpy array with shape (..., 3)
        voxels = np.floor(
            (points - self.bounds.lower_bound) / self.resolution
        ).astype(np.int32)
        return voxels

    def voxel_to_point(self, voxels):
        if type(voxels) == list:
            voxels = np.array(voxels)
        # The voxels is in numpy array with shape (..., 3)
        # The points is in numpy array with shape (..., 3)
        points = voxels * self.resolution + self.bounds.lower_bound
        return points

    def voxel_to_index(self, voxels):
        if type(voxels) == list:
            voxels = np.array(voxels)
        # The voxels is in numpy array with shape (..., 3)
        # The indexex is in numpy array with shape (...,)
        indexes = (
            voxels[..., 0] * self.voxel_num[1] * self.voxel_num[2]
            + voxels[..., 1] * self.voxel_num[2]
            + voxels[..., 2]
        )
        return indexes

    def index_to_voxel(self, indexes):
        if type(indexes) == list:
            indexes = np.array(indexes)
        # The indexes is in numpy array with shape (...,)
        # The voxels is in numpy array with shape (..., 3)
        voxels = np.zeros((indexes.shape + (3,)), dtype=np.int32)
        voxels[..., 2] = indexes % self.voxel_num[2]
        indexes = indexes // self.voxel_num[2]
        voxels[..., 1] = indexes % self.voxel_num[1]
        voxels[..., 0] = indexes // self.voxel_num[1]
        return voxels

    def point_to_index(self, points):
        # The points is in numpy array with shape (..., 3)
        # The indexes is in numpy array with shape (...,)
        voxels = self.point_to_voxel(points)
        indexes = self.voxel_to_index(voxels)
        return indexes

    def index_to_point(self, indexes):
        # The indexes is in numpy array with shape (...,)
        # The points is in numpy array with shape (..., 3)
        voxels = self.index_to_voxel(indexes)
        points = self.voxel_to_point(voxels)
        return points

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

    def index_to_pcd(self, indexes) -> o3d.visualization.draw_geometries:
        points = self.index_to_point(indexes)
        colos = np.array([self.indexes_colors_mapping_dict[idx] for idx in indexes])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colos)
        return pcd
