from sg.controller import Controller
from sg.utils.utils import vis_depth
from sg.utils.utils import get_inlier_mask
from sg.scripts.realsense_recorder import RecorderImage
import argparse
import time
import numpy as np
import open3d as o3d
from sg.scripts.rgb_feature_match import RGBFeatureMatch

def build_sg(args):

    controller = Controller(
        step=0, 
        tags=args.tags, 
        interval=3, 
        resolution=0.01,
        occ_avoid_radius=0.2,
        save_memory=args.save_memory,
        debug=args.debug,
        serial=args.rs_serial_number
    )

    if args.scanning_room:
        # data collection and pose estimation, if you have data, please don't use it
        # to avoid delete exist data
        controller.data_collection()
        return

    if args.preprocess:
        controller.pose_estimation()

        # show droid-slam pose pointcloud
        controller.show_droidslam_pointcloud(use_inlier_mask=False, is_visualize=False)

        # transform droid-slam pose to floor base coord
        controller.transform_pose_with_floor(display_result=False)

        # use transformed pose train ace for relocalize
        controller.train_ace()

        # vis_depth(controller.recorder_dir)
   
        controller.show_pointcloud(is_visualize=False)

    # when first times, init scenario
    controller.get_view_dataset()
    controller.get_semantic_memory()
    controller.get_instances()
    controller.get_instance_scene_graph()
    controller.get_lightglue_features()

    controller.show_pointcloud()
    print("""
        press "B" to show background
        press "C" to color by class
        press "R" to color by rgb
        press "F" to color by clip sim
        press "G" to toggle scene graph
        press "I" to color by instance
        press "O" to toggle bbox
        press "V" to save view params""")
    controller.show_instances(
        controller.instance_objects, 
        clip_vis=True, 
        scene_graph=controller.instance_scene_graph, 
        show_background=True
    )
    controller.instance_scene_graph.export_json("./scene_graph.json")

    return controller

def update_sg(controller: Controller, imagerecorder: RecorderImage):
    # get observation
    points, colors, depths, mask = imagerecorder.get_observations()
    colors = (colors / 255.0)[:, :, ::-1]
    depths = depths * imagerecorder.depth_scale
    observations = {}
    observations[0] = {
        "point": points,
        "rgb": colors,
        "depth": depths,
        "mask": mask,
        #"c2b": imagerecorder._get_cam2base(),
        "intrinsic": imagerecorder.intrinsic_matrix,
        "dist_coef": imagerecorder.dist_coef,
    }
    pcds = []
    for name, obs in observations.items():
        # Get the rgb, point cloud, and the camera pose
        color = obs["rgb"]
        point = obs["point"]
        mask = obs["mask"]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point[mask])
        pcd.colors = o3d.utility.Vector3dVector(color[mask])
        pcds.append(pcd)

    o3d.visualization.draw_geometries(pcds)

    # get pose
    for name, obs in observations.items():
        point = obs["point"]
        rgb = obs["rgb"]
        mask = obs["mask"]
        inlier_mask = get_inlier_mask(point=point, color=rgb, mask=mask)
        mask = np.logical_and(mask, inlier_mask)
        obs["mask"] = mask
    rough_poses = controller.test_ace(observations)
    for index, (name, obs) in enumerate(observations.items()):
        obs["pose"], obs["pose_inlier_num"] = rough_poses[index]
    featurematch = RGBFeatureMatch()
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

        target_index, best_matches_len = featurematch.find_most_similar_image(
            image, features=controller.lightglue_features, visualize=False, view_dataset=controller.view_dataset)
        target_rgb = (controller.view_dataset.images[target_index] / 255).astype(np.float32)
        target_point_w = controller.view_dataset.global_points[target_index]
        target_mask = controller.view_dataset.masks[target_index]
        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(target_point_w[target_mask])
        target_pcd.colors = o3d.utility.Vector3dVector(target_rgb[target_mask])
        target_pcds += target_pcd

    is_success, ref_tf_matrix = controller.calculate_alignment_colored_icp(source_pcds, target_pcds)
    if is_success:
        for name, obs in observations.items():
            obs["pose"] = np.dot(ref_tf_matrix, obs["pose"])
        controller.show_pointcloud_for_align(observations)
        controller.update_scene(observations=observations)
        controller.show_instances(
            controller.instance_objects, 
            clip_vis=True, 
            scene_graph=controller.instance_scene_graph, 
            show_background=True
        )
        controller.instance_scene_graph.export_json("./scene_graph.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='demo of sg.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--tags', type=str, default="room1", help='tags for scene.')
    parser.add_argument('--save_memory', type=bool, default=True, help='save each step memory.')

    parser.add_argument('--scanning_room', action='store_true', help='For hand camera to recorder scene.')
    parser.add_argument('--preprocess', action='store_true', help='preprocess scene.')
    parser.add_argument('--debug', action='store_true', help='For debug mode.')

    parser.add_argument('--task_scene_change_level', type=str, default="Minor Adjustment", 
                        choices=["Minor Adjustment", "Positional Shift", "Appearance"], help='scene change level.')
    parser.add_argument('--rs_serial_number', type=str, default="", help='Intel Realsense device serial number.')


    args = parser.parse_args()

    controller = build_sg(args)
    if not args.scanning_room:
        imagerecorder = RecorderImage(serial_number=args.rs_serial_number)
        while True:
            ans = input("Press y to capture the updated scene picture. Press q to quit.")
            if ans == "y":
                update_sg(controller, imagerecorder)
            elif ans == "q":
                break
