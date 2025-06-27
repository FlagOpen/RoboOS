import json
import numpy as np
from pathlib import Path
from collections.abc import Iterable
import torch
import torch.nn.functional as F
import numpy as np
import open3d as o3d
import pickle
from tqdm import tqdm, trange
from PIL import Image
import cv2
from typing import Union
from collections import Counter
import faiss
import sys
import time
from typing import List, Tuple

from sg.memory.instances.instance_utils import DetectionList, MapObjectList
from sg.memory.instances.instance_utils import to_tensor, to_numpy, get_bbox
from sg.memory.view_dataset import ViewDataset
# from dovisg.utils.instance_utils import load_result
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import copy

class InstanceProcess:
    def __init__(
        self,
        downsample_voxel_size: float=0.01,
        part_level_classes: list=["handle"]
    ):
        self.downsample_voxel_size = downsample_voxel_size

        self.mask_area_threshold = 25
        self.max_bbox_area_ratio = 0.3
        self.mask_conf_threshold = 0.25
        self.min_points_threshold = 10

        self.dbscan_remove_noise = True
        self.dbscan_eps = self.downsample_voxel_size * 2  # 0.1
        # dbscan_min_points = 10
        # self.min_samples_ratio = 0.05
        self.min_samples = 10
        self.sample_rate = 0.95

        # self.eps = self.downsample_voxel_size * 4
        self.expend_eps = self.downsample_voxel_size * 2

        self.spatial_weight = 0.5
        self.vis_weight = 0.4
        self.text_weight = 0.1
        self.sim_threshold = 0.75
        assert self.spatial_weight + self.vis_weight + self.text_weight == 1
        assert 0 < self.sim_threshold < 1

        # Perform post-processing periodically if told so
        self.denoise_interval = 20

        self.merge_overlap_thresh = 0.90
        self.merge_visual_sim_thresh = 0.95
        self.merge_text_sim_thresh = 0.9

        self.part_level_classes = part_level_classes

    def get_object_id(self, class_name):
        if class_name not in self.class_id_counts:
            self.class_id_counts[class_name] = 0
        else:
            self.class_id_counts[class_name] += 1
        return f"{class_name}_{self.class_id_counts[class_name]}"

    def get_instances(
        self,
        memory_dir,
        view_dataset: ViewDataset,
        # classes,
        objects: Union[MapObjectList, None]=None,
        obj_min_detections=3,
    ):

        # The indexes that are initially identified as objects cannot 
        # be used as background indexes to avoid affecting the occupancy map.
        self.objects_original_indexes = []

        self.view_dataset = view_dataset
        append_length = self.view_dataset.append_length_log[-1]
        pixel_index_mappings = self.view_dataset.pixel_index_mappings[-append_length:]
        pixel_index_masks = self.view_dataset.pixel_index_masks[-append_length:]
        names = self.view_dataset.names[-append_length:]

        self.denoise_interval = max(self.denoise_interval, int(append_length / 5))

        self.class_id_counts = {}

        if objects is not None:
            objects, self.class_id_counts = self.load_objects(objects, self.class_id_counts)
        else:
            objects = MapObjectList()

        for idx in tqdm(range(len(names)), desc="instance process"):
            # image_original_pil = Image.fromarray(images[idx])
            name = names[idx]
            # load grounded SAM 2 detections
            gsam2_obs = None # stands for grounded SAM 2 observations
            detections_path = memory_dir / f"semantic_memory" / f"{name}.pkl"
            with open(detections_path, "rb") as f:
                gsam2_obs = pickle.load(f)

            fg_detection_list = self.gsam2_obs_to_detection_list(
                gsam2_obs=gsam2_obs,
                pixel_indexes=pixel_index_mappings[idx],
                pixel_indexes_mask=pixel_index_masks[idx],
                # class_names=classes,
                image_name=name,
            )
            
            if len(fg_detection_list) == 0:
                continue

            if len(objects) == 0:
                # Add all detections to the map
                for i in range(len(fg_detection_list)):
                    objects.append(fg_detection_list[i])

                # Skip the similarity computation 
                continue
            
            spatial_sim = self.compute_spatial_similarities(fg_detection_list, objects)
            visual_sim = self.compute_visual_similarities(fg_detection_list, objects)
            text_sim = self.compute_text_similarities(fg_detection_list, objects)
            agg_sim = self.aggregate_similarities(spatial_sim, visual_sim, text_sim)

            # Threshold sims according to sim_threshold. Set to negative infinity if below threshold
            agg_sim[agg_sim < self.sim_threshold] = float('-inf')

            objects = self.merge_detections_to_objects(fg_detection_list, objects, agg_sim)

            if (idx+1) % self.denoise_interval == 0:
                objects = self.denoise_objects(objects)

        print("====> denoise objects")
        objects = self.denoise_objects(objects)
        print("merge objects")
        objects = self.merge_objects(objects)

        # Make each indexes only belonging to one instance-level object
        # using twice filter to filter invalid objects
        objects = self.filter_objects(objects, obj_min_detections=obj_min_detections)
        objects = self.indexes_align_objects(objects)
        objects = self.filter_objects(objects, obj_min_detections=obj_min_detections)
        

        # torch to numpy and delete bbox
        objects = self.change_objects(objects)
        

        objects_original_indexes = np.unique(self.objects_original_indexes) 
        objects_indexes = np.concatenate(objects.get_values("indexes"))
        # when at step 1, bottom is true
        # assert len(np.intersect1d(objects_original_indexes, objects_indexes)) == len(objects_indexes)
        object_filter_indexes = np.setdiff1d(objects_original_indexes, objects_indexes)

        # instance-level objects
        return objects, object_filter_indexes  
    
    
    
    def indexes_align_objects(self, objects: MapObjectList):
        index_to_object = {}
        # object_class_names = objects.get_most_common_class_name()
        # object_class_confidences = np.array(objects.get_most_common_class_conf())
        object_class_confidences = np.array([obj["conf"] for obj in objects])
        for cnt, obj in enumerate(tqdm(objects, total=len(objects), desc="align index to object")):
             # if object class is part level lebels, don't filter it for easy find parent object
            # if object_class_names[cnt] not in self.part_level_classes: 
            if obj["class_name"] not in self.part_level_classes:
                for index in obj["indexes"]:
                    if index not in index_to_object:
                        index_to_object[index] = []
                    index_to_object[index].append(cnt)
        
        # filter indexes which exist in more than two object
        filtered_index_to_object = dict((k, v) for k, v in index_to_object.items() if len(v) > 1)
        for index, obj_idxes in tqdm(filtered_index_to_object.items(),
                total=len(filtered_index_to_object), desc="processing filtered indexes"):
            confs = object_class_confidences[obj_idxes]
            max_conf_idx = np.argmax(confs)
            # remove index in object when confidence is not biggest
            for cnt in obj_idxes:
                if cnt != max_conf_idx:
                    objects[cnt]["indexes"].remove(index)

        return objects


    def change_objects(self, objects: MapObjectList):
        for obj in objects:
            # It will change later, and the amount of calculation is very small
            obj['clip_ft'] = to_numpy(obj['clip_ft'])
            obj['text_ft'] = to_numpy(obj['text_ft'])
            del obj["bbox"]
        return objects
    
    def load_objects(self, objects: MapObjectList, class_id_counts: dict):
        for obj in objects:
            # It will change later, and the amount of calculation is very small
            obj['clip_ft'] = to_tensor(obj['clip_ft'])
            obj['text_ft'] = to_tensor(obj['text_ft'])
            obj["bbox"] = self.get_bounding_box(obj["indexes"])
            label_count = int(obj["class_id"].split("_")[1])
            if obj["class_name"] not in class_id_counts.keys():
                class_id_counts[obj["class_name"]] = int(obj["class_id"].split("_")[1])
            else:
                if class_id_counts[obj["class_name"]] < label_count:
                    class_id_counts[obj["class_name"]] = label_count

        return objects, class_id_counts

    # def filter_objects(self, objects: MapObjectList, obj_min_points=3, obj_min_detections=3):
    def filter_objects(self, objects: MapObjectList, obj_min_detections=3):
        # Remove the object that has very few points or viewed too few times
        print("Before filtering:", len(objects))
        objects_to_keep = []
        for obj in tqdm(objects, total=len(objects), desc="filter objects"):
            # if len(obj['indexes']) >= obj_min_points and obj['num_detections'] >= obj_min_detections:
            if len(obj['indexes']) >= self.min_points_threshold and obj['num_detections'] >= obj_min_detections:
                objects_to_keep.append(obj)
        objects = MapObjectList(objects_to_keep)
        print("After filtering:", len(objects))
        return objects

    def merge_objects(self, objects: MapObjectList):
        if self.merge_overlap_thresh > 0:
            start_time = time.time()
            # Merge one object into another if the former is contained in the latter
            overlap_matrix = self.compute_overlap_matrix(objects)
            print("Before merging:", len(objects))
            objects = self.merge_overlap_objects(objects, overlap_matrix)
            print("After merging:", len(objects))
            print(f"merge time spend {time.time() - start_time}")
        
        return objects


    def compute_overlap_matrix(self, objects: MapObjectList):
        '''
        compute pairwise overlapping between objects in terms of point nearest neighbor. 
        Suppose we have a list of n point cloud, each of which is a o3d.geometry.PointCloud object. 
        Now we want to construct a matrix of size n x n, where the (i, j) entry is the ratio of points in point cloud i 
        that are within a distance threshold of any point in point cloud j. 
        '''
        n = len(objects)
        overlap_matrix = np.zeros((n, n))
        
        # Convert the point clouds into numpy arrays and then into FAISS indices for efficient search
        point_arrays = [self.view_dataset.index_to_point(obj['indexes']) for obj in objects]
        indices = [faiss.IndexFlatL2(arr.shape[1]) for arr in point_arrays]
        
        # Add the points from the numpy arrays to the corresponding FAISS indices
        for index, arr in zip(indices, point_arrays):
            index.add(arr)

        # Compute the pairwise overlaps
        for i in range(n):
            for j in range(n):
                if i != j:  # Skip diagonal elements
                    box_i = objects[i]['bbox']
                    box_j = objects[j]['bbox']
                    
                    # Skip if the boxes do not overlap at all (saves computation)
                    iou = self.compute_3d_iou(box_i, box_j)
                    if iou == 0:
                        continue
                    
                    # # Use range_search to find points within the threshold
                    # _, I = indices[j].range_search(point_arrays[i], threshold ** 2)
                    D, I = indices[j].search(point_arrays[i], 1)

                    # # If any points are found within the threshold, increase overlap count
                    # overlap += sum([len(i) for i in I])

                    overlap = (D < self.downsample_voxel_size ** 2).sum() # D is the squared distance

                    # Calculate the ratio of points within the threshold
                    overlap_matrix[i, j] = overlap / len(point_arrays[i])

        return overlap_matrix

    def merge_detections_to_objects(
        self,
        detection_list: DetectionList, 
        objects: MapObjectList, 
        agg_sim: torch.Tensor
    ) -> MapObjectList:
        # Iterate through all detections and merge them into objects
        for i in range(agg_sim.shape[0]):
            # If not matched to any object, add it as a new object
            if agg_sim[i].max() == float('-inf'):
                objects.append(detection_list[i])
            # Merge with most similar existing object
            else:
                j = agg_sim[i].argmax()
                matched_det = detection_list[i]
                matched_obj = objects[j]
                merged_obj = self.merge_obj_to_obj(matched_obj, matched_det, run_dbscan=False)
                objects[j] = merged_obj
                
        return objects

    def merge_overlap_objects(self, objects: MapObjectList, overlap_matrix: np.ndarray):
        x, y = overlap_matrix.nonzero()
        overlap_ratio = overlap_matrix[x, y]

        sort = np.argsort(overlap_ratio)[::-1]
        x = x[sort]
        y = y[sort]
        overlap_ratio = overlap_ratio[sort]

        kept_objects = np.ones(len(objects), dtype=bool)
        for i, j, ratio in zip(x, y, overlap_ratio):
            visual_sim = F.cosine_similarity(
                to_tensor(objects[i]['clip_ft']),
                to_tensor(objects[j]['clip_ft']),
                dim=0
            )
            text_sim = F.cosine_similarity(
                to_tensor(objects[i]['text_ft']),
                to_tensor(objects[j]['text_ft']),
                dim=0
            )
            print(objects[i]["class_name"], objects[j]["class_name"], ratio, visual_sim, text_sim)

            # Use stricter methods to judge
            if ratio * self.spatial_weight + visual_sim * self.vis_weight + text_sim * self.text_weight > self.sim_threshold + 0.05:
            # if ratio > self.merge_overlap_thresh:
            #     if visual_sim > self.merge_visual_sim_thresh and \
            #         text_sim > self.merge_text_sim_thresh:
                    if kept_objects[j]:
                        # Then merge object i into object j
                        objects[j] = self.merge_obj_to_obj(objects[j], objects[i], run_dbscan=True)
                        kept_objects[i] = False
            else:
                break
    
        # Remove the objects that have been merged
        new_objects = [obj for obj, keep in zip(objects, kept_objects) if keep]
        objects = MapObjectList(new_objects)
        return objects


    def denoise_objects(self, objects: MapObjectList):
        for i in tqdm(range(len(objects)), total=len(objects), desc="denoise objects"):
            og_object_indexes = objects[i]['indexes']
            objects[i]['indexes'] = self.process_indexes(objects[i]['indexes'], run_dbscan=True)
            if len(objects[i]['indexes']) < 4:
                objects[i]['indexes'] = og_object_indexes
                continue
            objects[i]['bbox'] = self.get_bounding_box(objects[i]['indexes'])
            objects[i]['bbox'].color = [0,1,0]
        return objects


    def compute_3d_iou(self, bbox1, bbox2, padding=0, use_iou=True):
        # Get the coordinates of the first bounding box
        bbox1_min = np.asarray(bbox1.get_min_bound()) - padding
        bbox1_max = np.asarray(bbox1.get_max_bound()) + padding

        # Get the coordinates of the second bounding box
        bbox2_min = np.asarray(bbox2.get_min_bound()) - padding
        bbox2_max = np.asarray(bbox2.get_max_bound()) + padding

        # Compute the overlap between the two bounding boxes
        overlap_min = np.maximum(bbox1_min, bbox2_min)
        overlap_max = np.minimum(bbox1_max, bbox2_max)
        overlap_size = np.maximum(overlap_max - overlap_min, 0.0)

        overlap_volume = np.prod(overlap_size)
        bbox1_volume = np.prod(bbox1_max - bbox1_min)
        bbox2_volume = np.prod(bbox2_max - bbox2_min)
        
        obj_1_overlap = overlap_volume / bbox1_volume
        obj_2_overlap = overlap_volume / bbox2_volume
        max_overlap = max(obj_1_overlap, obj_2_overlap)

        iou = overlap_volume / (bbox1_volume + bbox2_volume - overlap_volume)

        if use_iou:
            return iou
        else:
            return max_overlap

    def compute_iou_batch(self, bbox1: torch.Tensor, bbox2: torch.Tensor) -> torch.Tensor:
        '''
        Compute IoU between two sets of axis-aligned 3D bounding boxes.
        
        bbox1: (M, V, D), e.g. (M, 8, 3)
        bbox2: (N, V, D), e.g. (N, 8, 3)
        
        returns: (M, N)
        '''
        # Compute min and max for each box
        bbox1_min, _ = bbox1.min(dim=1) # Shape: (M, 3)
        bbox1_max, _ = bbox1.max(dim=1) # Shape: (M, 3)
        bbox2_min, _ = bbox2.min(dim=1) # Shape: (N, 3)
        bbox2_max, _ = bbox2.max(dim=1) # Shape: (N, 3)

        # Expand dimensions for broadcasting
        bbox1_min = bbox1_min.unsqueeze(1)  # Shape: (M, 1, 3)
        bbox1_max = bbox1_max.unsqueeze(1)  # Shape: (M, 1, 3)
        bbox2_min = bbox2_min.unsqueeze(0)  # Shape: (1, N, 3)
        bbox2_max = bbox2_max.unsqueeze(0)  # Shape: (1, N, 3)

        # Compute max of min values and min of max values
        # to obtain the coordinates of intersection box.
        inter_min = torch.max(bbox1_min, bbox2_min)  # Shape: (M, N, 3)
        inter_max = torch.min(bbox1_max, bbox2_max)  # Shape: (M, N, 3)

        # Compute volume of intersection box
        inter_vol = torch.prod(torch.clamp(inter_max - inter_min, min=0), dim=2)  # Shape: (M, N)

        # Compute volumes of the two sets of boxes
        bbox1_vol = torch.prod(bbox1_max - bbox1_min, dim=2)  # Shape: (M, 1)
        bbox2_vol = torch.prod(bbox2_max - bbox2_min, dim=2)  # Shape: (1, N)

        # Compute IoU, handling the special case where there is no intersection
        # by setting the intersection volume to 0.
        iou = inter_vol / (bbox1_vol + bbox2_vol - inter_vol + 1e-10)

        return iou

    def expand_3d_box(self, bbox: torch.Tensor) -> torch.Tensor:
        '''
        Expand the side of 3D boxes such that each side has at least expend_eps length.
        Assumes the bbox cornder order in open3d convention. 
        
        bbox: (N, 8, D)
        
        returns: (N, 8, D)
        '''
        center = bbox.mean(dim=1)  # shape: (N, D)

        va = bbox[:, 1, :] - bbox[:, 0, :]  # shape: (N, D)
        vb = bbox[:, 2, :] - bbox[:, 0, :]  # shape: (N, D)
        vc = bbox[:, 3, :] - bbox[:, 0, :]  # shape: (N, D)
        
        a = torch.linalg.vector_norm(va, ord=2, dim=1, keepdim=True)  # shape: (N, 1)
        b = torch.linalg.vector_norm(vb, ord=2, dim=1, keepdim=True)  # shape: (N, 1)
        c = torch.linalg.vector_norm(vc, ord=2, dim=1, keepdim=True)  # shape: (N, 1)
        
        va = torch.where(a < self.expend_eps, va / a * self.expend_eps, va)  # shape: (N, D)
        vb = torch.where(b < self.expend_eps, vb / b * self.expend_eps, vb)  # shape: (N, D)
        vc = torch.where(c < self.expend_eps, vc / c * self.expend_eps, vc)  # shape: (N, D)
        
        new_bbox = torch.stack([
            center - va/2.0 - vb/2.0 - vc/2.0,
            center + va/2.0 - vb/2.0 - vc/2.0,
            center - va/2.0 + vb/2.0 - vc/2.0,
            center - va/2.0 - vb/2.0 + vc/2.0,
            center + va/2.0 + vb/2.0 + vc/2.0,
            center - va/2.0 + vb/2.0 + vc/2.0,
            center + va/2.0 - vb/2.0 + vc/2.0,
            center + va/2.0 + vb/2.0 - vc/2.0,
        ], dim=1) # shape: (N, 8, D)
        
        new_bbox = new_bbox.to(bbox.device)
        new_bbox = new_bbox.type(bbox.dtype)
        
        return new_bbox

    def compute_3d_iou_accuracte_batch(self, bbox1, bbox2):
        '''
        Compute IoU between two sets of oriented (or axis-aligned) 3D bounding boxes.
        
        bbox1: (M, 8, D), e.g. (M, 8, 3)
        bbox2: (N, 8, D), e.g. (N, 8, 3)
        
        returns: (M, N)
        '''
        # Must expend the box beforehand, otherwise it may results overestimated results
        bbox1 = self.expand_3d_box(bbox1)
        bbox2 = self.expand_3d_box(bbox2)
        import pytorch3d.ops as ops
        bbox1 = bbox1[:, [0, 2, 5, 3, 1, 7, 4, 6]]
        bbox2 = bbox2[:, [0, 2, 5, 3, 1, 7, 4, 6]]
        inter_vol, iou = ops.box3d_overlap(bbox1.float(), bbox2.float())
        return iou

    def compute_overlap_matrix_2set(self, objects_map: MapObjectList, objects_new: DetectionList,
                                    bbox_map: torch.Tensor, bbox_new: torch.Tensor) -> np.ndarray:
        '''
        compute pairwise overlapping between two set of objects in terms of point nearest neighbor. 
        objects_map is the existing objects in the map, objects_new is the new objects to be added to the map
        Suppose len(objects_map) = m, len(objects_new) = n
        Then we want to construct a matrix of size m x n, where the (i, j) entry is the ratio of points 
        in point cloud i that are within a distance threshold of any point in point cloud j.
        '''
        m = len(objects_map)
        n = len(objects_new)
        overlap_matrix = np.zeros((m, n))
        
        # Convert the point clouds into numpy arrays and then into FAISS indices for efficient search
        # points_map = [np.asarray(obj['pcd'].points, dtype=np.float32) for obj in objects_map] # m arrays
        points_map = [self.view_dataset.index_to_point(obj['indexes']) for obj in objects_map]
        indices_map = [faiss.IndexFlatL2(arr.shape[1]) for arr in points_map] # m indices
        
        # Add the points from the numpy arrays to the corresponding FAISS indices
        for index, arr in zip(indices_map, points_map):
            index.add(arr)
            
        # points_new = [np.asarray(obj['pcd'].points, dtype=np.float32) for obj in objects_new] # n arrays
        points_new = [self.view_dataset.index_to_point(obj['indexes']) for obj in objects_new]
            
        # bbox_map = objects_map.get_stacked_values_torch('bbox')
        # bbox_new = objects_new.get_stacked_values_torch('bbox')
        try:
            iou = self.compute_3d_iou_accuracte_batch(bbox_map, bbox_new) # (m, n)
        except ValueError:
            print("Met `Plane vertices are not coplanar` error, use axis aligned bounding box instead")
            bbox_map = []
            bbox_new = []
            # for pcd in objects_map.get_values('pcd'):
            #     bbox_map.append(np.asarray(
            #         pcd.get_axis_aligned_bounding_box().get_box_points()))
                
            # for pcd in objects_new.get_values('pcd'):
            #     bbox_new.append(np.asarray(
            #         pcd.get_axis_aligned_bounding_box().get_box_points()))

            for indexes in objects_map.get_values('indexes'):
                pcd = self.view_dataset.index_to_pcd(indexes=indexes)
                bbox_map.append(np.asarray(
                    pcd.get_axis_aligned_bounding_box().get_box_points()))
                
            for indexes in objects_new.get_values('indexes'):
                pcd = self.view_dataset.index_to_pcd(indexes=indexes)
                bbox_new.append(np.asarray(
                    pcd.get_axis_aligned_bounding_box().get_box_points()))
                
            bbox_map = torch.from_numpy(np.stack(bbox_map))
            bbox_new = torch.from_numpy(np.stack(bbox_new))
            
            iou = self.compute_iou_batch(bbox_map, bbox_new) # (m, n)
                

        # Compute the pairwise overlaps
        for i in range(m):
            for j in range(n):
                if iou[i, j] < 1e-6:
                    continue
                
                D, I = indices_map[i].search(points_new[j], 1) # search new object j in map object i

                overlap = (D < self.downsample_voxel_size ** 2).sum() # D is the squared distance

                # Calculate the ratio of points within the threshold
                overlap_matrix[i, j] = overlap / len(points_new[j])

        return overlap_matrix


    def compute_spatial_similarities(self, detection_list: DetectionList, objects: MapObjectList) -> torch.Tensor:
        '''
        Compute the spatial similarities between the detections and the objects
        
        Args:
            detection_list: a list of M detections
            objects: a list of N objects in the map
        Returns:
            A MxN tensor of spatial similarities
        '''
        det_bboxes = detection_list.get_stacked_values_torch('bbox')
        obj_bboxes = objects.get_stacked_values_torch('bbox')

        spatial_sim = self.compute_overlap_matrix_2set(objects, detection_list, obj_bboxes, det_bboxes)
        spatial_sim = torch.from_numpy(spatial_sim).T

        return spatial_sim

    def compute_visual_similarities(self, detection_list: DetectionList, objects: MapObjectList) -> torch.Tensor:
        '''
        Compute the visual similarities between the detections and the objects
        
        Args:
            detection_list: a list of M detections
            objects: a list of N objects in the map
        Returns:
            A MxN tensor of visual similarities
        '''
        det_fts = detection_list.get_stacked_values_torch('clip_ft') # (M, D)
        obj_fts = objects.get_stacked_values_torch('clip_ft') # (N, D)

        det_fts = det_fts.unsqueeze(-1) # (M, D, 1)
        obj_fts = obj_fts.T.unsqueeze(0) # (1, D, N)
        
        visual_sim = F.cosine_similarity(det_fts, obj_fts, dim=1) # (M, N)
        
        return visual_sim

    def compute_text_similarities(self, detection_list: DetectionList, objects: MapObjectList) -> torch.Tensor:
        '''
        Compute the text similarities between the detections and the objects
        
        Args:
            detection_list: a list of M detections
            objects: a list of N objects in the map
        Returns:
            A MxN tensor of text similarities
        '''
        det_fts = detection_list.get_stacked_values_torch('text_ft') # (M, D)
        obj_fts = objects.get_stacked_values_torch('text_ft') # (N, D)

        det_fts = det_fts.unsqueeze(-1) # (M, D, 1)
        obj_fts = obj_fts.T.unsqueeze(0) # (1, D, N)
        
        text_sim = F.cosine_similarity(det_fts, obj_fts, dim=1) # (M, N)
        
        return text_sim

    def aggregate_similarities(self, spatial_sim: torch.Tensor, visual_sim: torch.Tensor, text_sim: torch.Tensor) -> torch.Tensor:
        '''
        Aggregate spatial and visual similarities into a single similarity score
        
        Args:
            spatial_sim: a MxN tensor of spatial similarities
            visual_sim: a MxN tensor of visual similarities
            text_sim: a MxN tenser of text_similarities
        Returns:
            A MxN tensor of aggregated similarities
        '''
        # sims = (1 + self.phys_bias) * spatial_sim + (1 - self.phys_bias) * visual_sim # (M, N)
        sims = spatial_sim * self.spatial_weight + visual_sim * self.vis_weight + text_sim * self.text_weight
        return sims

    def resize_gsam2_obs(
        self,
        gsam2_obs,
        HW,
    ):
        n_masks = len(gsam2_obs['xyxy'])

        new_mask = []
        
        for mask_idx in range(n_masks):
            mask = gsam2_obs['mask'][mask_idx]
            if mask.shape != HW:
                # Rescale the xyxy coordinates to the image shape
                x1, y1, x2, y2 = gsam2_obs['xyxy'][mask_idx]
                x1 = round(x1 * HW[1] / mask.shape[1])
                y1 = round(y1 * HW[0] / mask.shape[0])
                x2 = round(x2 * HW[1] / mask.shape[1])
                y2 = round(y2 * HW[0] / mask.shape[0])
                gsam2_obs['xyxy'][mask_idx] = [x1, y1, x2, y2]
                
                # Reshape the mask to the image shape
                mask = cv2.resize(mask.astype(np.uint8), HW[::-1], interpolation=cv2.INTER_NEAREST)
                mask = mask.astype(bool)
                new_mask.append(mask)

        if len(new_mask) > 0:
            gsam2_obs['mask'] = np.asarray(new_mask)
            
        return gsam2_obs

    def filter_gsam2_obs(
        self,
        gsam2_obs: dict,
        HW,
    ):
        # If no detection at all
        if len(gsam2_obs['xyxy']) == 0:
            return gsam2_obs
        
        # Filter out the objects based on various criteria
        idx_to_keep = []
        for mask_idx in range(len(gsam2_obs['xyxy'])):
            local_class_id = gsam2_obs['class_id'][mask_idx]
            class_name = gsam2_obs['classes'][local_class_id]
            
            # SKip masks that are too small
            if gsam2_obs['mask'][mask_idx].sum() < max(self.mask_area_threshold, 10):
                continue
            
            # Skip the boxes that are too large
            image_area = HW[0] * HW[1]
            mask_area = gsam2_obs['mask'][mask_idx].sum()
            if mask_area > self.max_bbox_area_ratio * image_area:
                # print(f"Skipping {class_name} with area {mask_area} > {self.max_bbox_area_ratio} * {image_area}")
                continue
                
            # Skip masks with low confidence
            if gsam2_obs['confidence'] is not None:
                if gsam2_obs['confidence'][mask_idx] < self.mask_conf_threshold:
                    continue
            
            idx_to_keep.append(mask_idx)
        
        for k in gsam2_obs.keys():
            if isinstance(gsam2_obs[k], str) or k == "classes": # Captions
                continue
            elif isinstance(gsam2_obs[k], list):
                gsam2_obs[k] = [gsam2_obs[k][i] for i in idx_to_keep]
            elif isinstance(gsam2_obs[k], np.ndarray):
                gsam2_obs[k] = gsam2_obs[k][idx_to_keep]
            else:
                raise NotImplementedError(f"Unhandled type {type(gsam2_obs[k])}")
        
        return gsam2_obs


    def mask_subtract_contained(self, xyxy: np.ndarray, mask: np.ndarray, th1=0.8, th2=0.7):
        '''
        Compute the containing relationship between all pair of bounding boxes.
        For each mask, subtract the mask of bounding boxes that are contained by it.
        
        Args:
            xyxy: (N, 4), in (x1, y1, x2, y2) format
            mask: (N, H, W), binary mask
            th1: float, threshold for computing intersection over box1
            th2: float, threshold for computing intersection over box2
            
        Returns:
            mask_sub: (N, H, W), binary mask
        '''
        N = xyxy.shape[0] # number of boxes

        # Get areas of each xyxy
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1]) # (N,)

        # Compute intersection boxes
        lt = np.maximum(xyxy[:, None, :2], xyxy[None, :, :2])  # left-top points (N, N, 2)
        rb = np.minimum(xyxy[:, None, 2:], xyxy[None, :, 2:])  # right-bottom points (N, N, 2)
        
        inter = (rb - lt).clip(min=0)  # intersection sizes (dx, dy), if no overlap, clamp to zero (N, N, 2)

        # Compute areas of intersection boxes
        inter_areas = inter[:, :, 0] * inter[:, :, 1] # (N, N)
        
        inter_over_box1 = inter_areas / areas[:, None] # (N, N)
        # inter_over_box2 = inter_areas / areas[None, :] # (N, N)
        inter_over_box2 = inter_over_box1.T # (N, N)
        
        # if the intersection area is smaller than th2 of the area of box1, 
        # and the intersection area is larger than th1 of the area of box2,
        # then box2 is considered contained by box1
        contained = (inter_over_box1 < th2) & (inter_over_box2 > th1) # (N, N)
        contained_idx = contained.nonzero() # (num_contained, 2)

        mask_sub = mask.copy() # (N, H, W)
        # mask_sub[contained_idx[0]] = mask_sub[contained_idx[0]] & (~mask_sub[contained_idx[1]])
        for i in range(len(contained_idx[0])):
            mask_sub[contained_idx[0][i]] = mask_sub[contained_idx[0][i]] & (~mask_sub[contained_idx[1][i]])

        return mask_sub


    def create_object(
        self, pixel_indexes, det_mask, pixel_indexes_mask
    ) -> Tuple[np.ndarray]:
        mask_real = np.logical_and(det_mask, pixel_indexes_mask)
        if mask_real.sum() == 0:
            indexes = np.array([])
        else:
            indexes = np.unique(pixel_indexes[mask_real])
        return indexes
    

    def indexes_denoise_dbscan(
        self,
        indexes: Union[np.ndarray, list],
    ):
        # min_samples=10
        # sample_rate=0.95

        if len(indexes) < self.min_samples:
            return indexes

        points = self.view_dataset.index_to_point(indexes)
        
        # pcd = self.view_dataset.index_to_pcd(indexes)
        # o3d.visualization.draw_geometries([pcd])

        neighbors = NearestNeighbors(n_neighbors=self.min_samples)
        neighbors_fit = neighbors.fit(points)
        distances, indices = neighbors_fit.kneighbors(points)
        distances = np.sort(distances[:, -1], axis=0)
        eps = distances[int(len(distances) * self.sample_rate)]

        db = DBSCAN(eps=eps, min_samples=self.min_samples).fit(points)
        labels = db.labels_

        # Count all labels in the cluster
        counter = Counter(labels)

        # Remove the noise label
        if counter and (-1 in counter):
            del counter[-1]

        if counter:
            # Find the label of the largest cluster
            most_common_label, _ = counter.most_common(1)[0]
            
            # Create mask for points in the largest cluster
            largest_mask = labels == most_common_label

            # Apply mask
            largest_cluster_indexes = np.array(indexes)[largest_mask]
            
            # If the largest cluster is too small, return the original point cloud
            if len(largest_cluster_indexes) < 5:
                return indexes

            # Create a new PointCloud object
            indexes = largest_cluster_indexes

        else:
            print(f"cluster error and pcd.points lenght is {len(indexes)}")
            
        return indexes

    def process_indexes(self, indexes, run_dbscan=True):
        indexes = np.unique(indexes)
        if self.dbscan_remove_noise and run_dbscan:
            indexes = self.indexes_denoise_dbscan(indexes)
        return list(indexes)

    def merge_obj_to_obj(self, obj1, obj2, run_dbscan=True):
        '''
        Merge the new object to the old object
        using the name of conf max one

        just save all result to obj1

        This operation is done in-place
        '''
        n_obj1_det = obj1['num_detections']
        n_obj2_det = obj2['num_detections']

        for k in obj1.keys():
            if k in ['caption']:
                # Here we need to merge two dictionaries and adjust the key of the second one
                for k2, v2 in obj2['caption'].items():
                    obj1['caption'][k2 + n_obj1_det] = v2
            elif k not in ['bbox', 'clip_ft', "text_ft", 'conf', 'class_name', 'class_id']:
                if isinstance(obj1[k], list) or isinstance(obj1[k], int):
                    obj1[k] += obj2[k]
                elif k == "inst_color":
                    obj1[k] = obj1[k] # Keep the initial instance color
                else:
                    raise NotImplementedError
            else: # bbox, clip_ft, text_ft, conf class_name, class_id are handled below
                continue
        
        if obj1["conf"] < obj2["conf"]:
            obj1["conf"] = obj2["conf"]
            obj1["class_name"] = obj2["class_name"]
            obj1["class_id"] = obj2["class_id"]

        obj1['indexes'] = self.process_indexes(obj1['indexes'], run_dbscan=run_dbscan)
        obj1['bbox'] = self.get_bounding_box(obj1['indexes'])
        obj1['bbox'].color = [0,1,0]
        
        # merge clip ft
        obj1['clip_ft'] = (obj1['clip_ft'] * n_obj1_det +
                        obj2['clip_ft'] * n_obj2_det) / (
                        n_obj1_det + n_obj2_det)
        obj1['clip_ft'] = F.normalize(obj1['clip_ft'], dim=0)
        
        obj2['text_ft'] = to_tensor(obj2['text_ft'])
        obj1['text_ft'] = to_tensor(obj1['text_ft'])
        obj1['text_ft'] = (obj1['text_ft'] * n_obj1_det +
                        obj2['text_ft'] * n_obj2_det) / (
                        n_obj1_det + n_obj2_det)
        obj1['text_ft'] = F.normalize(obj1['text_ft'], dim=0)
        
        return obj1


    def get_bounding_box(self, indexes):
        pcd = self.view_dataset.index_to_pcd(indexes=indexes)
        bbox = get_bbox(pcd)
        return bbox

    def gsam2_obs_to_detection_list(
        self,
        gsam2_obs: dict, 
        pixel_indexes: np.ndarray,
        pixel_indexes_mask: np.ndarray,
        # class_names: str=None,
        image_name: str= None,
    ):
        '''
        Return a DetectionList object from the gobs
        All object are still in the camera frame. 
        '''
        HW = pixel_indexes.shape[:2]
        fg_detection_list = DetectionList()

        gsam2_obs = self.resize_gsam2_obs(gsam2_obs, HW)
        gsam2_obs = self.filter_gsam2_obs(gsam2_obs, HW)

        if len(gsam2_obs['xyxy']) == 0:
            return fg_detection_list
        
        # Compute the containing relationship among all detections and subtract fg from bg objects
        xyxy = gsam2_obs['xyxy']
        det_masks = gsam2_obs['mask']
        gsam2_obs['mask'] = self.mask_subtract_contained(xyxy, det_masks)

        n_masks = len(gsam2_obs['xyxy'])
        for mask_idx in range(n_masks):
            local_class_id = gsam2_obs['class_id'][mask_idx]
            det_mask = gsam2_obs['mask'][mask_idx]
            class_name = gsam2_obs['classes'][local_class_id]
            # global_class_id = -1 if class_names is None else class_names.index(class_name)

            # global_object_pcd = self.create_object_pcd(
            #     pixel_indexes, det_mask, pixel_indexes_mask
            # )
            
            global_object_indexes = self.create_object(
                pixel_indexes, det_mask, pixel_indexes_mask
            )
            self.objects_original_indexes.extend(global_object_indexes.tolist())
            # It at least contains 5 points
            if len(global_object_indexes) < max(self.min_points_threshold, 5): 
                continue

            # get largest cluster, filter out noise 
            # global_object_pcd = self.process_pcd(global_object_pcd)
            global_object_indexes = self.process_indexes(global_object_indexes)

            pcd_bbox = self.get_bounding_box(global_object_indexes)
            pcd_bbox.color = [0,1,0]

            if pcd_bbox.volume() < 1e-6:
                continue
            
            class_id = self.get_object_id(class_name=class_name)
            # Treat the detection in the same way as a 3D object
            # Store information that is enough to recover the detection
            detected_object = {
                'mask_idx' : [mask_idx],                         # idx of the mask/detection
                'image_name' : [image_name],                     # path to the RGB image

                # not list
                'class_name' : class_name,                         # most conf class name for this object
                'class_id': class_id,                      # for scene graph
                'conf': gsam2_obs['confidence'][mask_idx],         # for scene graph

                # 'class_id' : [global_class_id],                         # global class id for this detection
                'num_detections' : 1,                            # number of detections in this object
                'mask': [det_mask],
                'xyxy': [gsam2_obs['xyxy'][mask_idx]],
                
                # 'n_points': [len(global_object_pcd.points)],
                # 'n_points': len(global_object_indexes),
                # 'pixel_area': [det_mask.sum()],
                # 'contain_number': [None],                          # This will be computed later
                "inst_color": np.random.rand(3),                 # A random color used for this segment instance
                # 'is_background': False,
                
                # These are for the entire 3D object
                # 'pcd': global_object_pcd,
                'indexes': global_object_indexes,
                'bbox': pcd_bbox,
                'clip_ft': to_tensor(gsam2_obs['image_feats'][mask_idx]),
                'text_ft': to_tensor(gsam2_obs['text_feats'][mask_idx]),
            }
            
            fg_detection_list.append(detected_object)
        
        return fg_detection_list