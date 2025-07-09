import os
import numpy as np
from sg.memory.view_dataset import ViewDataset
from sg.memory.instances.instance_utils import MapObjectList
from sg.memory.instances.instance_utils import get_bbox
from sg.memory.scene_graph.graph import ObjectNode, SceneGraph
import alphashape
from shapely.geometry import Polygon, MultiPolygon
import copy
import open3d as o3d
from tqdm import tqdm
from typing import Union
import faiss
from scipy.spatial import ConvexHull, Delaunay

def _get_cross_prod_mat(pVec_Arr):
    # pVec_Arr shape (3)
    qCross_prod_mat = np.array(
        [
            [0, -pVec_Arr[2], pVec_Arr[1]],
            [pVec_Arr[2], 0, -pVec_Arr[0]],
            [-pVec_Arr[1], pVec_Arr[0], 0],
        ]
    )
    return qCross_prod_mat

def _caculate_align_mat(pVec_Arr):
    scale = np.linalg.norm(pVec_Arr)
    pVec_Arr = pVec_Arr / scale
    # must ensure pVec_Arr is also a unit vec.
    z_unit_Arr = np.array([0, 0, 1])
    z_mat = _get_cross_prod_mat(z_unit_Arr)

    z_c_vec = np.matmul(z_mat, pVec_Arr)
    z_c_vec_mat = _get_cross_prod_mat(z_c_vec)
    if np.dot(z_unit_Arr, pVec_Arr) == -1:
        qTrans_Mat = -np.eye(3, 3)
    elif np.dot(z_unit_Arr, pVec_Arr) == 1:
        qTrans_Mat = np.eye(3, 3)
    else:
        qTrans_Mat = (
            np.eye(3, 3)
            + z_c_vec_mat
            + np.matmul(z_c_vec_mat, z_c_vec_mat) / (1 + np.dot(z_unit_Arr, pVec_Arr))
        )
    qTrans_Mat *= scale
    return qTrans_Mat


# Borrow ideas and codes from H. Sánchez's answer
# https://stackoverflow.com/questions/59026581/create-arrows-in-open3d
def getArrowMesh(origin=[0, 0, 0], end=None, color=[0, 0, 0]):
    vec_Arr = np.array(end) - np.array(origin)
    vec_len = np.linalg.norm(vec_Arr)
    # mesh_arrow = o3d.geometry.TriangleMesh.create_arrow(
    #     cone_height=0.2 * vec_len,
    #     cone_radius=0.08,
    #     cylinder_height=1.0 * vec_len,
    #     cylinder_radius=0.04,
    # )
    mesh_arrow = o3d.geometry.TriangleMesh.create_arrow(
        cone_height=0.02 * vec_len,
        cone_radius=0.008,
        cylinder_height=0.10 * vec_len,
        cylinder_radius=0.004,
    )
    mesh_arrow.paint_uniform_color(color)
    rot_mat = _caculate_align_mat(vec_Arr / vec_len)
    mesh_arrow.rotate(rot_mat, center=np.array([0, 0, 0]))
    mesh_arrow.translate(np.array(origin))
    return mesh_arrow

class SceneGraphProcesser:
    def __init__(
        self,
        part_level_classes: list,
        resolution: float=0.01,
        neighbour_num: int=5,
        stand_floor_threshold: float=0.15,
        alphashape_alpha: float=1,
        alpha_shape_overleaf_rate_threshold: float=0.6,
        part_intersection_rate_threshold: float=0.2,
        inside_threshold: float=0.95
    ):
        self.part_level_classes = part_level_classes
        self.resolution = resolution
        self.neighbour_num = neighbour_num
        self.stand_floor_threshold = stand_floor_threshold
        self.alphashape_alpha = alphashape_alpha
        self.alpha_shape_overleaf_rate_threshold = alpha_shape_overleaf_rate_threshold
        self.part_intersection_rate_threshold = part_intersection_rate_threshold
        self.inside_threshold = inside_threshold
        
        self.root_node_class = "floor"
        self.root_node_id = "floor_0"


    def get_voxel_neighbours(self, voxels, size):
        offsets = np.arange(-size, size + 1)
        # Create a 3x3x3 grid of offsets
        di, dj, dk = np.meshgrid(offsets, offsets, offsets, indexing="ij")
        # Flatten the grids and stack them to create a list of offset vectors
        offset_vectors = np.stack((di.ravel(), dj.ravel(), dk.ravel()), axis=-1)
        # Remove the (0,0,0) offset (central voxel)
        offset_vectors = offset_vectors[~np.all(offset_vectors == 0, axis=1)]
        # Apply the offsets to the voxel coordinates using broadcasting
        neighbours = voxels[:, np.newaxis, :] + offset_vectors
        valid_mask = np.all((neighbours >= 0) & (neighbours < self.view_dataset.voxel_num), axis=2)
        return self.view_dataset.voxel_to_index(neighbours), valid_mask

    def get_handle_info(
        self, 
        instance_object, 
        parent_instance_object, 
        sibling_instance_objects, 
        visualize=False
    ):
        # Get the handle pose

        # handle_pcd = instance.index_to_pcd(instance.voxel_indexes)
        voxel_indexes = instance_object["indexes"]
        handle_point = self.view_dataset.index_to_point(voxel_indexes)

        # Currently the handle points are kind of clean
        handle_center = np.mean(handle_point, axis=0)
        # PCA to get the handle most obvious direction
        centralized_points = handle_point - handle_center
        cov_matrix = np.cov(centralized_points, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(
            cov_matrix + np.eye(cov_matrix.shape[0]) * 1e-10
        )
        handle_direction = eigenvectors[:, -1]
        floor_normal = np.array([0, 0, 1])
        # Make the handle direciton prependicular or parallel to the floor
        if np.abs(np.dot(handle_direction, floor_normal)) > 0.7:
            handle_direction = floor_normal
        else:
            handle_direction = np.cross(
                floor_normal, np.cross(handle_direction, floor_normal)
            )
            handle_direction /= np.linalg.norm(handle_direction)

        # Get the neightbour points of the handle
        
        voxels = self.view_dataset.index_to_voxel(instance_object["indexes"])
        neighbours, valid_mask = self.get_voxel_neighbours(voxels, self.neighbour_num)

        # Only consider the neighbours on the parent instance
        # valid_mask *= np.isin(neighbours, list(parent_instance.voxel_indexes))
        valid_mask *= np.isin(neighbours, parent_instance_object["indexes"])

        all_neighbour_indexes = np.array(list(set(neighbours[valid_mask].tolist())))
        all_neighbour_indexes = all_neighbour_indexes[
            ~np.isin(all_neighbour_indexes, voxel_indexes)
        ]
        # neighbour_pcd = self.index_to_pcd(all_neighbour_indexes).tolist()
        neighbour_point = self.view_dataset.index_to_point(all_neighbour_indexes).tolist()
        
        # Estimate the normals
        neighbour_pc = o3d.geometry.PointCloud()

        # neighbour_pc.points = o3d.utility.Vector3dVector(neighbour_pcd)
        neighbour_pc.points = o3d.utility.Vector3dVector(neighbour_point)

        neighbour_pc.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.08, max_nn=20)
        )
        neighbour_normals = np.asarray(neighbour_pc.normals)
        rounded_normals = np.round(neighbour_normals, decimals=3)
        # Remove the normals prependicular to the floor
        mask = np.abs(np.dot(rounded_normals, floor_normal)) < 0.7
        rounded_normals = rounded_normals[mask]
        # Remove the normals parallel to the handle direction
        mask = np.abs(np.dot(rounded_normals, handle_direction)) < 0.7
        rounded_normals = rounded_normals[mask]
        unique, counts = np.unique(rounded_normals, axis=0, return_counts=True)
        open_direction = unique[np.argmax(counts)]
        # Use heuristic to determine the reference direction
        reference_direction = np.zeros(3) - handle_center
        reference_direction /= np.linalg.norm(reference_direction)
        if np.dot(open_direction, reference_direction) < 0:
            open_direction = -open_direction
        # Make the open direction perpendicular to the [0, 0, 1]
        open_direction = np.cross(floor_normal, np.cross(open_direction, floor_normal))
        open_direction /= np.linalg.norm(open_direction)
        # Refine the handle direction using the open direction
        # Use heuristic to judge the joint type, if vertical, revolute, or prismatic
        if np.abs(np.dot(handle_direction, floor_normal)) < 0.7:
            handle_direction = np.cross(open_direction, floor_normal)
            joint_type = "prismatic"
        else:
            joint_type = "revolute"

        joint_info = {}
        if joint_type == "revolute":
            joint_axis = handle_direction
            # Get the side_direction
            side_direction = np.cross(handle_direction, open_direction)
            # Get the distance of the handle center to the parent node boundary in the side direction
            # parent_points = parent_instance.index_to_pcd(parent_instance.voxel_indexes)

            parent_points = self.view_dataset.index_to_point(parent_instance_object["indexes"])

            # Get the distance between the handle center to the parent node boundary
            distances = np.dot(parent_points - handle_center, side_direction)
            max_distance = np.max(distances)
            min_distance = np.min(distances)
            # Judge if there is another handle on the side
            flag_current = False
            flag_reverse = False
            # for sibling_instance in sibling_instances:
            for sibling_instance_object in sibling_instance_objects:
                if sibling_instance_object["class_id"] == "handle":
                    distance_up = np.dot(
                        np.mean(
                            self.view_dataset.index_to_point(
                                sibling_instance_object["indexes"]
                            ),
                            axis=0,
                        )
                        - handle_center,
                        floor_normal,
                    )
                    if np.abs(distance_up) > 0.07:
                        continue
                    distance_current = np.dot(
                        np.mean(
                            self.view_dataset.index_to_point(
                                sibling_instance_object["indexes"]
                            ),
                            axis=0,
                        )
                        - handle_center,
                        side_direction,
                    )
                    if distance_current > 0:
                        flag_current = True
                    distance_reverse = np.dot(
                        np.mean(
                            self.view_dataset.index_to_point(
                                sibling_instance_object["indexes"]
                            ),
                            axis=0,
                        )
                        - handle_center,
                        -side_direction,
                    )
                    if distance_reverse > 0:
                        flag_reverse = True

            if flag_current:
                side_direction = -side_direction
                joint_axis = -joint_axis
            elif not flag_reverse and max_distance < np.abs(min_distance):
                side_direction = -side_direction
                joint_axis = -joint_axis

            # Find the points in the farthest open directions
            open_distances = np.dot(parent_points, open_direction)
            max_distance = np.dot(handle_center, open_direction)
            potential_points = parent_points[
                np.logical_and(
                    open_distances < max_distance, open_distances > max_distance - 0.05
                )
            ]
            # Find the points in the farthest side directions
            side_distances = np.dot(potential_points, side_direction)
            max_index = np.argmax(side_distances)
            joint_info["side_direction"] = side_direction
            joint_info["joint_origin"] = potential_points[max_index]
            joint_info["joint_axis"] = joint_axis     

        if visualize:
            # scene_pcd, scene_color = self.view_dataset.indexes_colors_mapping_dict
            # # Test the estimated normal of current scen first
            # scene_pc = o3d.geometry.PointCloud()
            # scene_pc.points = o3d.utility.Vector3dVector(scene_pcd)
            # scene_pc.colors = o3d.utility.Vector3dVector(scene_color)
            # o3d.visualization.draw_geometries(
            #     [scene_pc, neighbour_pc], point_show_normal=True
            # )

            scene_pc = self.view_dataset.index_to_pcd(list(self.view_dataset.indexes_colors_mapping_dict.keys()))

            arrow1 = getArrowMesh(
                origin=handle_center,
                end=handle_center + handle_direction,
                color=[1, 0, 0],
            )
            arrow2 = getArrowMesh(
                origin=handle_center,
                end=handle_center + open_direction,
                color=[0, 1, 0],
            )
            extra = [arrow1, arrow2]
            if joint_type == "revolute":  # revolute表示旋转关节，Prismatic表示平移关节
                arrow3 = getArrowMesh(
                    origin=joint_info["joint_origin"],
                    end=joint_info["joint_origin"] + joint_info["joint_axis"],
                    color=[0, 0, 0],
                )
                extra.append(arrow3)
            # visualize_pc(scene_pcd, color=scene_color, extra=extra)
            o3d.visualization.draw_geometries(
                [scene_pc] + extra, point_show_normal=True
            )

        return handle_center, handle_direction, open_direction, joint_type, joint_info

    def build_scene_graph(self, view_dataset: ViewDataset, instance_objects: MapObjectList, instance_scene_graph: Union[SceneGraph, None]=None):
        self.view_dataset = view_dataset

        # use floor node as root node
        # floor node is not real exist, we can assume it as z-axis is 0
        # self.root_node_class = "floor"
        # self.root_node_id = "floor_0"
        if instance_scene_graph is None:
            root_node = ObjectNode(
                parent=None,
                node_class=self.root_node_class,
                node_id=self.root_node_id
            )
            instance_scene_graph = SceneGraph(root_node=root_node)


        class_id_to_instance_object = {ins_obj["class_id"]: ins_obj for ins_obj in instance_objects}


        """At first, we process non-part level object"""
        # just include non-part object
        label_to_object_information_mapping = {}
        for idx, ins_obj in tqdm(enumerate(instance_objects), 
                total=len(instance_objects), desc="label non-part object information mapping"):
            class_name = ins_obj["class_name"]
            # At first, don't process part level labels
            if class_name in self.part_level_classes:
                continue

            class_id = ins_obj["class_id"]
            
            lower_z = np.min(self.view_dataset.index_to_point(ins_obj["indexes"]), axis=0)[2]
            xy_axis_unique = np.unique(self.view_dataset.index_to_point(ins_obj["indexes"])[:, :2], axis=0)
            # Perturb the points a bit to avoid colinearity
            xy_axis_unique += np.random.normal(0, 4e-3, xy_axis_unique.shape)
            alpha_shape = alphashape.alphashape(xy_axis_unique, alpha=self.alphashape_alpha)

            if isinstance(alpha_shape, MultiPolygon):
                alpha_shape = max(alpha_shape.geoms, key=lambda shape: shape.area)
            
            label_to_object_information_mapping[class_id] = {
                "ins_obj": ins_obj,
                "lower_z": lower_z,
                "alpha_shape": alpha_shape
            }

        # sorted by lower z-axis value for improve efficiency
        label_to_object_information_mapping = dict(
            sorted(
                label_to_object_information_mapping.items(),
                key=lambda x: x[1]["lower_z"])
            )

        for class_id, info in label_to_object_information_mapping.items():
            if class_id not in instance_scene_graph.object_nodes.keys():
                if info["lower_z"] < self.stand_floor_threshold:
                    ins_obj = info["ins_obj"]
                    assert ins_obj["class_id"] == class_id
                    instance_scene_graph.add_node(
                        parent=instance_scene_graph.root,
                        node_class=ins_obj["class_name"],
                        node_id=ins_obj["class_id"],
                        parent_relation="on",
                        is_part=False
                    )

        need_align_class_id_list = [class_id for class_id in label_to_object_information_mapping.keys()
                                          if class_id not in instance_scene_graph.object_nodes.keys()]
        
        # For other objects not directly on the table, find the object beneath it
        while len(need_align_class_id_list) > 0:
            for class_id in need_align_class_id_list:
                info = label_to_object_information_mapping[class_id]
                ins_obj = info["ins_obj"]
                lower_z = info["lower_z"]
                alpha_shape = info["alpha_shape"]
                bottom_ins_objs = []
                for other_class_id, other_info in label_to_object_information_mapping.items():
                    if class_id != other_class_id:
                        other_ins_obj = other_info["ins_obj"]
                        other_lower_z = other_info["lower_z"]
                        other_alpha_shape = other_info["alpha_shape"]

                        inter_rate = alpha_shape.intersection(other_alpha_shape).area / alpha_shape.area \
                              if alpha_shape.area > 0 else 0

                        if inter_rate > self.alpha_shape_overleaf_rate_threshold and other_lower_z < lower_z:
                            print(f"{class_id} {other_class_id}, overlap rate {inter_rate}, \
                                  bottom distance:{lower_z - other_lower_z}")
                            
                            bottom_ins_objs.append([other_ins_obj, lower_z - other_lower_z])

                if len(bottom_ins_objs) > 0:
                    bottom_ins_obj = sorted(bottom_ins_objs, key=lambda x: x[1])[0][0]
                    # if bottom object has not been add to scene graph, continue it
                    if bottom_ins_obj["class_id"] not in instance_scene_graph.object_nodes.keys():
                        continue
                    
                    parent_node = instance_scene_graph.object_nodes[bottom_ins_obj["class_id"]]
                    assert ins_obj["class_id"] == class_id
                    instance_scene_graph.add_node(
                        parent=parent_node,
                        node_class=ins_obj["class_name"],
                        node_id=ins_obj["class_id"],
                        parent_relation="on",
                        is_part=False
                    )
                else: # That is, if the object has no other objects at the bottom, it still belongs to the floor
                    instance_scene_graph.add_node(
                        instance_scene_graph.root,
                        node_class=ins_obj["class_name"],
                        node_id=ins_obj["class_id"],
                        parent_relation="on",
                        is_part=False
                    )
                need_align_class_id_list.remove(class_id)


        """Now, we process part level object"""
        part_instance_objects = [ins_obj for ins_obj in instance_objects 
                                 if ins_obj["class_name"] in self.part_level_classes]
        

        def find_parent_instance_object(part_ins_obj):
            # Get the indexes of the part instance object
            part_ins_obj_indexes = part_ins_obj["indexes"]
            
            # Convert part object indexes into actual point coordinates
            part_points = self.view_dataset.index_to_point(part_ins_obj_indexes)
            
            # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
            # o3d.visualization.draw_geometries([self.view_dataset.index_to_pcd(part_ins_obj_indexes), coordinate_frame])

            # Initialize a FAISS L2 distance index for fast nearest neighbor search
            index = faiss.IndexFlatL2(part_points.shape[1])
            index.add(part_points)  # Add part object's points to the FAISS index

            parent_class_ids = []
            threshold = self.resolution * 10  # Define the distance threshold (e.g., 0.01 units)
            # Iterate through all the objects to find the best parent candidate
            for class_id, info in label_to_object_information_mapping.items():
                ins_obj = info["ins_obj"]
                ins_obj_indexes = ins_obj["indexes"]
                
                # Convert parent object indexes into point coordinates
                parent_points = self.view_dataset.index_to_point(ins_obj_indexes)
                
                # Use FAISS to search for the nearest points in the part object for each parent point
                D, I = index.search(parent_points, 1)  # D contains the distances, I contains the indices of nearest points

                # Calculate the ratio of points within the threshold distance
                close_points_rate = np.sum(D < threshold ** 2) / len(D)  # Compute the proportion of points below the threshold distance

                # If any points satisfy the distance condition, record the parent candidate
                if close_points_rate > 0:
                    parent_class_ids.append([class_id, close_points_rate])
            
            # If parent candidates are found, select the one with the highest proportion of close points
            if parent_class_ids:
                parent_class_id = sorted(parent_class_ids, key=lambda x: x[1], reverse=True)[0][0]
                return True, parent_class_id
            else:
                # Raise an error if no suitable parent object is found
                # raise ValueError("Cannot find a suitable parent object")
                return False, None
            
        no_parent_part_objects = []
        for part_ins_obj in tqdm(part_instance_objects, 
                total=len(part_instance_objects), desc="part-level object process"):
            # find parent objects
            find_flag, parent_class_id = find_parent_instance_object(part_ins_obj)
            if find_flag:
                instance_scene_graph.add_node(
                    instance_scene_graph.object_nodes[parent_class_id],
                    node_class=part_ins_obj["class_name"],
                    node_id=part_ins_obj["class_id"],
                    parent_relation="belong",
                    is_part=True
                )
            else:
                print("can't find handle object's parent.")
                no_parent_part_objects.append(part_ins_obj)
                coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
                o3d.visualization.draw_geometries([self.view_dataset.index_to_pcd(part_ins_obj["indexes"]), coordinate_frame])

        # delete part object which not find parent objects
        for no_parent_part_obj in no_parent_part_objects:
            part_instance_objects.remove(no_parent_part_obj)
            # label_to_object_information_mapping include non-part object's alpha-shape
            lower_z = np.min(self.view_dataset.index_to_point(no_parent_part_obj["indexes"]), axis=0)[2]
            nop_class_name = no_parent_part_obj["class_name"]
            nop_class_id = no_parent_part_obj["class_id"]
            if lower_z < self.stand_floor_threshold:
                instance_scene_graph.add_node(
                    parent=instance_scene_graph.root,
                    node_class=nop_class_name,
                    node_id=nop_class_id,
                    parent_relation="on",
                    is_part=False
                )
            else:
                xy_axis_unique = np.unique(self.view_dataset.index_to_point(no_parent_part_obj["indexes"])[:, :2], axis=0)
                # Perturb the points a bit to avoid colinearity
                xy_axis_unique += np.random.normal(0, 4e-3, xy_axis_unique.shape)
                alpha_shape = alphashape.alphashape(xy_axis_unique, alpha=self.alphashape_alpha)
                if isinstance(alpha_shape, MultiPolygon):
                    alpha_shape = max(alpha_shape.geoms, key=lambda shape: shape.area)

                bottom_ins_objs = []
                for other_class_id, other_info in label_to_object_information_mapping.items():
                    if nop_class_id != other_class_id:
                        other_ins_obj = other_info["ins_obj"]
                        other_lower_z = other_info["lower_z"]
                        other_alpha_shape = other_info["alpha_shape"]

                        inter_rate = alpha_shape.intersection(
                            other_alpha_shape
                        ).area / alpha_shape.area if alpha_shape.area > 0 else 0

                        if inter_rate > self.alpha_shape_overleaf_rate_threshold and other_lower_z < lower_z:
                            print(f"{nop_class_id} {other_class_id}, overlap rate {inter_rate}, \
                                    bottom distance:{lower_z - other_lower_z}")
                            
                            bottom_ins_objs.append([other_ins_obj, lower_z - other_lower_z])

                if len(bottom_ins_objs) > 0:
                    bottom_ins_obj = sorted(bottom_ins_objs, key=lambda x: x[1])[0][0]
                    parent_node = instance_scene_graph.object_nodes[bottom_ins_obj["class_id"]]
                    instance_scene_graph.add_node(
                        parent=parent_node,
                        node_class=nop_class_name,
                        node_id=nop_class_id,
                        parent_relation="on",
                        is_part=False
                    )
                else: # That is, if the object has no other objects at the bottom, it still belongs to the floor
                    instance_scene_graph.add_node(
                        instance_scene_graph.root,
                        node_class=nop_class_name,
                        node_id=nop_class_id,
                        parent_relation="on",
                        is_part=False
                    )

        # Analyze the attributes of the handles
        # for part_ins_obj in tqdm(part_instance_objects, total=len(part_instance_objects), desc="handle process"):
        #     if part_ins_obj["class_name"] != "handle":
        #         continue
            
        #     part_node = instance_scene_graph.object_nodes[part_ins_obj["class_id"]]
        #     parent_node = part_node.parent
        #     sibling_nodes = parent_node.children.values()

        #     parent_ins_obj = class_id_to_instance_object[parent_node.node_id]
        #     sibling_ins_objs = [class_id_to_instance_object[node.node_id] for node in sibling_nodes]

        #     (
        #         part_node.handle_center,
        #         part_node.handle_direction,
        #         part_node.open_direction,
        #         part_node.joint_type,
        #         joint_info,
        #     ) = self.get_handle_info(
        #         instance_object=part_ins_obj,
        #         parent_instance_object=parent_ins_obj,
        #         sibling_instance_objects=sibling_ins_objs,
        #         visualize=False,
        #     )

        #     if part_node.joint_type == "revolute":
        #         part_node.node_label = "door_handle"
        #         part_node.joint_axis = joint_info["joint_axis"]
        #         part_node.joint_origin = joint_info["joint_origin"]
        #         part_node.side_direction = joint_info["side_direction"]
        #     elif part_node.joint_type == "prismatic":
        #         part_node.node_label = "drawer_handle"
        

        # process `inside` relationship
        part_level_parent_nodes = set()
        for class_id, node in instance_scene_graph.object_nodes.items():
            if class_id == instance_scene_graph.root.node_id:
                continue
            ins_obj = class_id_to_instance_object[class_id]
            
            # Check if the node has a parent with part-level children
            parent_node = node.parent
            if parent_node is None:
                continue

            part_level_children = [
                child for child in parent_node.children.values() if child.is_part
            ]
            
            if not part_level_children:
                continue  # Skip if parent node does not have part-level objects
            
            part_level_parent_nodes.add(parent_node)

        for parent_node in part_level_parent_nodes:
            # Get the point cloud of the parent object
            parent_ins_obj = class_id_to_instance_object[parent_node.node_id]
            parent_points = self.view_dataset.index_to_point(parent_ins_obj["indexes"])

            delaunay_tri = Delaunay(parent_points)
            # all_descendants = collect_all_descendants(parent_node)

            if False:
                def extract_surface_triangles(delaunay):
                    triangles = []
                    for tetrahedron in delaunay.simplices:
                        triangles.append([tetrahedron[0], tetrahedron[1], tetrahedron[2]])
                        triangles.append([tetrahedron[0], tetrahedron[1], tetrahedron[3]])
                        triangles.append([tetrahedron[0], tetrahedron[2], tetrahedron[3]])
                        triangles.append([tetrahedron[1], tetrahedron[2], tetrahedron[3]])
                    return np.array(triangles)
                surface_triangles = extract_surface_triangles(delaunay_tri)
                parent_pcd = o3d.geometry.PointCloud()
                parent_pcd.points = o3d.utility.Vector3dVector(parent_points)
                parent_pcd.paint_uniform_color([121/255, 255/255, 226/255])
                triangles = o3d.utility.Vector3iVector(surface_triangles)
                triangle_mesh = o3d.geometry.TriangleMesh()
                triangle_mesh.vertices = o3d.utility.Vector3dVector(parent_points)
                triangle_mesh.triangles = triangles
                triangle_mesh.paint_uniform_color([121/255, 255/255, 226/255])
                o3d.visualization.draw_geometries([parent_pcd, triangle_mesh])


            # Check all descendant of the parent node (excluding part-level objects)
            # using delaunay to select the object that satisfies the requirements
            for child_node in parent_node.children.values():
                if child_node.is_part:
                    continue  # Skip part-level objects

                # Get the point cloud of the child object
                child_instance = class_id_to_instance_object[child_node.node_id]
                child_points = self.view_dataset.index_to_point(child_instance["indexes"])
                # just inner is valid
                inside_mask = delaunay_tri.find_simplex(child_points) > 0

                
                if inside_mask.sum() > self.inside_threshold * len(child_points):
                    # If the condition holds, set the parent_relationship to 'inside' and update parent
                    child_node.parent_relation = "inside"
                    print(f"Node {child_node.node_id} is inside {parent_node.node_id}")

        return instance_scene_graph
    
    def calculate_max_generation(self, node, current_generation=1):
        "find node children to get all affected node"
        max_generation = current_generation
        for child in list(node.children.values()):
            child_generation = self.calculate_max_generation(child, current_generation + 1)
            max_generation = max(max_generation, child_generation)
        return max_generation 

    def update_scene_graph(self, view_dataset: ViewDataset, instance_objects: MapObjectList, history_scene_graph: SceneGraph):
        class_id_to_instance_object = {ins_obj["class_id"]: ins_obj for ins_obj in instance_objects}
        last_step_class_ids = list(history_scene_graph.object_nodes.keys())
        # remove root class_id
        last_step_class_ids.remove(self.root_node_id)
        this_step_class_ids = list(class_id_to_instance_object.keys())
        del_class_ids = np.setdiff1d(last_step_class_ids, this_step_class_ids)
        new_class_ids = np.setdiff1d(this_step_class_ids, last_step_class_ids)

        affected_node_class_ids = []

        for del_class_id in del_class_ids:
            node = history_scene_graph.object_nodes[del_class_id]
            for object_node in history_scene_graph.object_nodes.values():
                if object_node.parent == node:
                    affected_node_class_ids.append(object_node.node_id)
        
        delete_node_class_ids = np.union1d(del_class_ids, affected_node_class_ids)

        # Sort by generation, the smaller the generation, the first to be deleted
        delete_node_generation_mapping = {}
        for del_class_id in delete_node_class_ids:
            node = history_scene_graph.object_nodes[del_class_id]
            delete_node_generation_mapping[del_class_id] = {
                "node": node,
                "generation": self.calculate_max_generation(node),
            }
            print(del_class_id, self.calculate_max_generation(node))

        delete_node_generation_mapping = dict(sorted(
            delete_node_generation_mapping.items(),
            key=lambda item: item[1]['generation']
        ))

        for key, value in delete_node_generation_mapping.items():
            object_node = value["node"]
            print(value["generation"], object_node.node_id)
            object_node.delete()
            del history_scene_graph.object_nodes[object_node.node_id]

        instance_scene_graph = self.build_scene_graph(
            view_dataset=view_dataset, 
            instance_objects=instance_objects, 
            instance_scene_graph=history_scene_graph
        )
        assert len(instance_scene_graph.object_nodes) == len(class_id_to_instance_object) + 1
        return instance_scene_graph