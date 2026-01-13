import sys
# sys.path.append('droid_slam')

from tqdm import tqdm
import numpy as np
import torch
import lietorch
import cv2
import os
import glob 
import time
import argparse
import json
from pathlib import Path
import shutil

droid_path = "../third_party/DROID-SLAM/droid_slam"
if droid_path not in sys.path:
    sys.path.append(droid_path)

from torch.multiprocessing import Process
from droid import Droid

import torch.nn.functional as F
import json
import quaternion

def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)

def image_stream(datadir, calib, stride, depth_ratio):
    """ image generator """

    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    colordir = os.path.join(datadir, "rgb")
    depthdir = os.path.join(datadir, "depth")

    color_list = sorted(os.listdir(colordir), key=lambda x: float(x.split(".")[0]))[::stride]
    depth_list = sorted(os.listdir(depthdir), key=lambda x: float(x.split(".")[0]))[::stride]

    for t, (colorfile, depthfile) in enumerate(zip(color_list, depth_list)):
        color = cv2.imread(os.path.join(colordir, colorfile))
        # depth = cv2.imread(os.path.join(depthdir, depthfile), -1)
        depth = np.load(os.path.join(depthdir, depthfile))
        if len(calib) > 4:
            color = cv2.undistort(color, K, calib[4:])

        h0, w0, _ = color.shape
        # just for image shape [1280, 600] from Realsense D455
        # h1 = int(h0 * np.sqrt((360 * 768) / (h0 * w0)))
        # w1 = int(w0 * np.sqrt((360 * 768) / (h0 * w0)))
        # h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
        # w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))
        h1 = int(h0 * np.sqrt((240 * 320) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((240 * 320) / (h0 * w0)))
        # h1 = int(h0 * np.sqrt((h0 * w0) / (h0 * w0)))
        # w1 = int(w0 * np.sqrt((h0 * w0) / (h0 * w0)))

        color = cv2.resize(color, (w1, h1))
        color = color[:h1-h1%8, :w1-w1%8]
        color = torch.as_tensor(color).permute(2, 0, 1)

        depth = depth.astype(np.float32) / depth_ratio
        #print(depth.sum() / depth.shape[0] / depth.shape[1])
        depth = torch.from_numpy(depth).float()[None,None]
        depth = F.interpolate(depth, (h1, w1), mode='nearest').squeeze()
        depth = depth[:h1-h1%8, :w1-w1%8]

        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        intrinsics[0::2] *= (w1 / w0)
        intrinsics[1::2] *= (h1 / h0)

        yield t, color[None], depth, intrinsics
        # yield t, color[None], intrinsics

def quaternion2transformation(quat):
    px, py, pz = quat[:3]
    qx, qy, qz, qw = quat[3:]
    q = quaternion.quaternion(qw, qx, qy, qz)
    rotation_matrix = quaternion.as_rotation_matrix(q)
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = [px, py, pz]
    return transformation_matrix

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=str, default="data_example/room1", help="path to image directory")
    parser.add_argument("--calib", type=str, default="data_example/room1/calib.txt", help="path to calibration file")
    parser.add_argument("--t0", default=0, type=int, help="starting frame")
    parser.add_argument("--stride", default=1, type=int, help="frame stride")

    parser.add_argument("--weights", default="checkpoints/droid-slam/droid.pth")
    parser.add_argument("--buffer", type=int, default=2048)
    # parser.add_argument("--image_size", default=[240, 320])
    parser.add_argument("--disable_vis", default=True, type=bool)

    parser.add_argument("--beta", type=float, default=0.3, help="weight for translation / rotation components of flow")
    parser.add_argument("--filter_thresh", type=float, default=2.4, help="how much motion before considering new keyframe")
    parser.add_argument("--warmup", type=int, default=8, help="number of warmup frames")
    parser.add_argument("--keyframe_thresh", type=float, default=4.0, help="threshold to create a new keyframe")
    parser.add_argument("--frontend_thresh", type=float, default=16.0, help="add edges between frames whithin this distance")
    parser.add_argument("--frontend_window", type=int, default=25, help="frontend optimization window")
    parser.add_argument("--frontend_radius", type=int, default=2, help="force edges between frames within radius")
    parser.add_argument("--frontend_nms", type=int, default=1, help="non-maximal supression of edges")

    parser.add_argument("--backend_thresh", type=float, default=22.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--pose_path", default="./poses_droidslam", help="path to saved reconstruction")
    parser.add_argument("--depth_ratio", type=float, default=1000.0)
    args = parser.parse_args()

    args.stereo = False
    torch.multiprocessing.set_start_method('spawn')

    droid = None

    # need high resolution depths
    if args.pose_path is not None:
        # args.reconstruction_path += "_" + str(time.time())
        args.upsample = True

    poses_dir = os.path.join(args.datadir, args.pose_path)
    if os.path.exists(poses_dir):
        shutil.rmtree(poses_dir)
    
    os.makedirs(poses_dir, exist_ok=True)

    tstamps = []
    image_gen = image_stream(args.datadir, args.calib, args.stride, args.depth_ratio)
    total_images = len(os.listdir(os.path.join(args.datadir, "rgb"))) // args.stride

    for (t, image, depth, intrinsics) in tqdm(image_gen, total=total_images, desc="Pose Estimation:"):
    # for (t, image, intrinsics) in tqdm(image_gen):
        if t < args.t0:
            continue

        if not args.disable_vis:
            show_image(image[0])

        if droid is None:
            args.image_size = [image.shape[2], image.shape[3]]
            droid = Droid(args)
        
        # droid.track(t, image, intrinsics=intrinsics)
        droid.track(t, image, depth=depth, intrinsics=intrinsics)

    traj_est = droid.terminate(image_stream(args.datadir, args.calib, args.stride, args.depth_ratio))
    print(f"Result Pose Number is {len(traj_est)}")
    # fill in poses for non-keyframes
    if args.pose_path is not None:
        os.makedirs(poses_dir, exist_ok=True)
        traj_dict = {}
        for i in range(len(traj_est)):
            T = quaternion2transformation(traj_est[i])
            np.savetxt(f"{poses_dir}/{i*args.stride:06}.txt", T)
