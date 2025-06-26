import copy
import matplotlib
import numpy as np
import pandas as pd
import open3d as o3d
import torch
import torch.nn.functional as F
from sg.perception.models.myclip import MyClip
from typing import Union
from sg.memory.view_dataset import ViewDataset
from sg.memory.instances.instance_utils import get_bbox, MapObjectList, LineMesh
from sg.memory.scene_graph.graph import SceneGraph

def normalized(a, axis=-1, order=2):
    """Normalizes a numpy array of points"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis), l2

# Copied from https://github.com/isl-org/Open3D/pull/738
def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector a to vector b with axis angle rotation
    """
    if np.array_equal(a, b):
        return None, None
    axis_ = np.cross(a, b)
    axis_ = axis_ / np.linalg.norm(axis_)
    angle = np.arccos(np.dot(a, b))

    return axis_, angle

def create_ball_mesh(center, radius, color=(0, 1, 0)):
    """
    Create a colored mesh sphere.
    
    Args:
    - center (tuple): (x, y, z) coordinates for the center of the sphere.
    - radius (float): Radius of the sphere.
    - color (tuple): RGB values in the range [0, 1] for the color of the sphere.
    
    Returns:
    - o3d.geometry.TriangleMesh: Colored mesh sphere.
    """
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    mesh_sphere.translate(center)
    mesh_sphere.paint_uniform_color(color)
    return mesh_sphere

def get_background_indexes(instance_objects: MapObjectList, view_dataset: ViewDataset):
    print("get background indexes.")
    all_instance_objects_indexes = np.concatenate(instance_objects.get_values("indexes"))
    # part-level indexes not be process, so below is not available
    # assert len(all_instance_objects_indexes) == len(np.unique(all_instance_objects_indexes))
    all_indexes = list(view_dataset.indexes_colors_mapping_dict.keys())
    # all_instance_objects_indexes must been subset of all_indexes
    assert len(np.intersect1d(all_indexes, all_instance_objects_indexes)) == len(np.unique(all_instance_objects_indexes))
    background_indexes = np.setdiff1d(all_indexes, all_instance_objects_indexes)
    return background_indexes


def calculate_max_generation(node, current_generation=1):
    "find node children to get all affected node"
    max_generation = current_generation
    for child in list(node.children.values()):
        child_generation = calculate_max_generation(child, current_generation + 1)
        max_generation = max(max_generation, child_generation)
    return max_generation 

def remove_extreme_points(points, percent=5):
    lower_bound = np.percentile(points, percent, axis=0)
    upper_bound = np.percentile(points, 100 - percent, axis=0)
    
    mask = (
        (points[:, 0] >= lower_bound[0]) & (points[:, 0] <= upper_bound[0]) & 
        (points[:, 1] >= lower_bound[1]) & (points[:, 1] <= upper_bound[1]) & 
        (points[:, 2] >= lower_bound[2]) & (points[:, 2] <= upper_bound[2])
    )
    
    return mask

def vis_instances(
        instance_objects: MapObjectList,
        class_colors: dict,
        view_dataset: ViewDataset,
        instance_scene_graph: Union[SceneGraph, None]=None,
        show_background: bool=False,
        clip_vis: bool=False,
        device="cuda",
        voxel_size: float=0.01
    ):
    if clip_vis:
        myclip = MyClip(device=device)
    cmap = matplotlib.colormaps.get_cmap("turbo")
    
    background_pcd_ = None
    object_pcds_ = []

    background_pcd = None
    if show_background:
        background_pcd = view_dataset.index_to_pcd(
            get_background_indexes(instance_objects, view_dataset)
        )
        background_pcd_ = copy.deepcopy(background_pcd)
        if True:
            # reduce noise pointcloud
            background_pcd, ind = background_pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.5)
            background_pcd = background_pcd.voxel_down_sample(voxel_size)

    # Sub-sample the point cloud for better interactive experience
    pcds = []
    bboxes = []
    object_classes = []


    for i in range(len(instance_objects)):

        if instance_objects[i]["class_id"] in ["doll_17", "plate_5"]:
            continue

        pcd = view_dataset.index_to_pcd(indexes=instance_objects[i]['indexes'])
        object_pcds_.append(pcd)
        pcd = pcd.voxel_down_sample(voxel_size)
        pcds.append(pcd)

        if True:  # for beautiful bbox
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors)
            mask = remove_extreme_points(points, percent=3)
            points = points[mask]
            colors = colors[mask]

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)

        object_classes.append(instance_objects[i]["class_name"])
        
        print(instance_objects[i]['class_id'])
        # o3d.visualization.draw_geometries([pcd])

        bbox = get_bbox(pcd)
        r = np.random.randint(0, 256)/255.0
        g = np.random.randint(0, 256)/255.0
        b = np.random.randint(0, 256)/255.0
        bbox.color = [r, g, b]
        

        bbox_points = np.asarray(bbox.get_box_points())


        bbox_lines = [
            [0, 1], [1, 7], [7, 2], [2, 0],
            [3, 5], [5, 4], [4, 6], [6, 3],
            [5, 2], [4, 7], [3, 0], [6, 1]
        ]

        line_mesh = LineMesh(
            bbox_points,
            bbox_lines,
            colors=bbox.color,
            radius=0.005
        )

        bboxes.append(line_mesh)


    pcds_copy = copy.deepcopy(pcds)

    if instance_scene_graph is not None:
        root_top= np.array([0, 0, -2])
        class_id_top_mapping = {}
        scene_graph_geometries = [
            # create_ball_mesh(root_top, 0.1, [1, 0, 0])
        ]
        for i in range(len(instance_objects)):
            ins_obj = instance_objects[i]
            point = view_dataset.index_to_point(ins_obj["indexes"])
            median_xy = np.median(point[:, :2], axis=0).tolist()
            max_z = np.max(point[:, 2]).item()
            ball_point = np.array(median_xy + [max_z])

            class_id = ins_obj["class_id"]
            node = instance_scene_graph.object_nodes[class_id]
            if len(node.children) == 0 and node.parent == instance_scene_graph.root:
                continue
            # extent = bboxes[i].get_max_bound()
            # extent = np.linalg.norm(extent)
            radius = 0.05
            # radius = extent ** 0.5 / 100
            class_id_top_mapping[ins_obj["class_id"]] = ball_point

            # remove the nodes on the ceiling, for better visualization
            ball = create_ball_mesh(ball_point, radius, class_colors[ins_obj["class_name"]])
            # ball = create_ball_mesh(center, radius, np.array([0, 0, 1]))
            scene_graph_geometries.append(ball)
        
        queue = [instance_scene_graph.root]
        while len(queue) > 0:
            node = queue.pop(0)
            for child in list(node.children.values()):
                if node.node_id != "floor_0":
                    node_top = class_id_top_mapping[node.node_id]

                    line_mesh = LineMesh(
                        points = np.array([
                            node_top, 
                            class_id_top_mapping[child.node_id]
                        ]),
                        lines = np.array([[0, 1]]),
                        colors = class_colors[child.node_id.split("_")[0]],
                        # colors = np.array([0, 0, 1]),
                        radius=0.01
                    )
                    scene_graph_geometries.extend(line_mesh.cylinder_segments)

                queue.append(child)

    # Set the title of the window
    vis = o3d.visualization.VisualizerWithKeyCallback()

    vis.create_window(window_name=f'Open3D', width=1280, height=720)

    for geometry in pcds:
        vis.add_geometry(geometry)
    

    vis_instances.show_scene_graph = True
    def toggle_scene_graph(vis):
        if instance_scene_graph is None:
            print("No instance scene graph provided")
            return

        if vis_instances.show_scene_graph:
            for geometry in scene_graph_geometries:
                vis.add_geometry(geometry, reset_bounding_box=False)
        else:
            for geometry in scene_graph_geometries:
                vis.remove_geometry(geometry, reset_bounding_box=False)
        
        vis_instances.show_scene_graph = not vis_instances.show_scene_graph


    vis_instances.show_bg_pcd = True
    def toggle_bg_pcd(vis):
        if background_pcd is None:
            print("No background objects found. Maybe you haven't set it.")
            return
        if vis_instances.show_bg_pcd:
            vis.add_geometry(background_pcd, reset_bounding_box=False)
        else:
            vis.remove_geometry(background_pcd, reset_bounding_box=False)
        vis_instances.show_bg_pcd = not vis_instances.show_bg_pcd


    def color_by_class(vis):
        for i in range(len(pcds)):
            pcd = pcds[i]
            obj_class = object_classes[i]
            pcd.colors = o3d.utility.Vector3dVector(
                np.tile(
                    class_colors[str(obj_class)],
                    (len(pcd.points), 1)
                )
            )

        for pcd in pcds:
            vis.update_geometry(pcd)

    def color_by_rgb(vis):
        for i in range(len(pcds)):
            pcd = pcds[i]
            pcd.colors = pcds_copy[i].colors
        
        for pcd in pcds:
            vis.update_geometry(pcd)
            
    def color_by_instance(vis):
        instance_colors = cmap(np.linspace(0, 1, len(pcds)))
        for i in range(len(pcds)):
            pcd = pcds[i]
            pcd.colors = o3d.utility.Vector3dVector(
                np.tile(
                    instance_colors[i, :3],
                    (len(pcd.points), 1)
                )
            )
            
        for pcd in pcds:
            vis.update_geometry(pcd)
        
    def color_by_clip_sim(vis):
        if not clip_vis:
            print("CLIP model is not initialized.")
            return
        
        text_query = input("Enter your query: ")
        text_queries = [text_query]
        text_query_ft = myclip.get_text_feature(text_queries)
        similarities = instance_objects.compute_similarities(text_query_ft).squeeze()

        max_value = similarities.max()
        min_value = similarities.min()
        normalized_similarities = (similarities - min_value) / (max_value - min_value)
        probs = F.softmax(similarities, dim=0)
        max_prob_idx = torch.argmax(probs)
        similarity_colors = cmap(normalized_similarities.detach().cpu().numpy())[..., :3]
        
        for i in range(len(instance_objects)):
            pcd = pcds[i]
            map_colors = np.asarray(pcd.colors)
            pcd.colors = o3d.utility.Vector3dVector(
                np.tile(
                    [
                        similarity_colors[i, 0].item(),
                        similarity_colors[i, 1].item(),
                        similarity_colors[i, 2].item()
                    ], 
                    (len(pcd.points), 1)
                )
            )

        for pcd in pcds:
            vis.update_geometry(pcd)


    vis_instances.show_bbox = True
    def toggle_bbox(vis):
        if vis_instances.show_bbox:
            for bbox in bboxes:
                for segment in bbox.cylinder_segments:
                    vis.add_geometry(segment, reset_bounding_box=False)
        else:
            for bbox in bboxes:
                for segment in bbox.cylinder_segments:
                    vis.remove_geometry(segment, reset_bounding_box=False)
        vis_instances.show_bbox = not vis_instances.show_bbox


    def save_view_params(vis):
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters("temp.json", param)
    
    vis.register_key_callback(ord("B"), toggle_bg_pcd)
    vis.register_key_callback(ord("C"), color_by_class)
    vis.register_key_callback(ord("R"), color_by_rgb)
    vis.register_key_callback(ord("F"), color_by_clip_sim)
    vis.register_key_callback(ord("G"), toggle_scene_graph)
    vis.register_key_callback(ord("I"), color_by_instance)
    vis.register_key_callback(ord("O"), toggle_bbox)
    vis.register_key_callback(ord("V"), save_view_params)
    
    # Render the scene
    vis.run()

    return object_pcds_ + [background_pcd_]