#!/usr/bin/env python3
# Copyright © Niantic, Inc. 2022.

import argparse
import logging
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from dataset import CamLocDataset

_logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description='Compute metrics for a pre-existing poses file.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('scene', type=Path, help='name of a scene in the dataset folder, e.g. Cambridge_GreatCourt')
    parser.add_argument('poses_file', type=Path, help='file containing poses estimated for the input scene')

    opt = parser.parse_args()

    # Load dataset.
    testset = CamLocDataset(
        opt.scene / "test",
        mode=0,  # Default for ACE, we don't need scene coordinates/RGB-D.
    )
    _logger.info(f"Loaded scene with {len(testset)} frames.")

    # load pre-existing poses
    with opt.poses_file.open('r') as f:
        frame_poses = f.readlines()
    _logger.info(f"Loaded {len(frame_poses)} poses.")

    # Check.
    assert len(testset) == len(frame_poses)

    # Keep track of rotation and translation errors for calculation of the median error.
    rErrs = []
    tErrs = []

    # Percentage of frames predicted within certain thresholds from their GT pose.
    pct10_5 = 0
    pct5 = 0
    pct2 = 0
    pct1 = 0

    # Iterate over the dataset.
    for image_idx, (_, _, gt_pose, _, _, _, _, image_file) in enumerate(testset):
        # Parse estimated pose from the input file.
        pose_file, qw, qx, qy, qz, tx, ty, tz, _, _, _ = frame_poses[image_idx].split()

        # Check that the files match.
        assert Path(image_file).name == pose_file

        # We do everything in np.
        gt_pose = gt_pose.numpy()

        # Convert quaternion to rotation matrix.
        r_mat = Rotation.from_quat([float(x) for x in [qx, qy, qz, qw]]).as_matrix()
        t_vec = np.array([float(x) for x in [tx, ty, tz]])

        # We saved the inverse pose
        estimated_pose_inv = np.eye(4)
        estimated_pose_inv[:3, :3] = r_mat
        estimated_pose_inv[:3, 3] = t_vec

        # Pose we use for evaluation.
        estimated_pose = np.linalg.inv(estimated_pose_inv)

        # calculate pose errors
        t_err = float(np.linalg.norm(gt_pose[0:3, 3] - estimated_pose[0:3, 3]))

        gt_R = gt_pose[0:3, 0:3]
        out_R = estimated_pose[0:3, 0:3]

        r_err = np.matmul(out_R, np.transpose(gt_R))
        r_err = cv2.Rodrigues(r_err)[0]
        r_err = np.linalg.norm(r_err) * 180 / math.pi

        _logger.info("Rotation Error: %.2fdeg, Translation Error: %.1fcm" % (r_err, t_err * 100))

        # Save the errors.
        rErrs.append(r_err)
        tErrs.append(t_err * 100)

        # Check various thresholds.
        if r_err < 5 and t_err < 0.1:  # 10cm/5deg
            pct10_5 += 1
        if r_err < 5 and t_err < 0.05:  # 5cm/5deg
            pct5 += 1
        if r_err < 2 and t_err < 0.02:  # 2cm/2deg
            pct2 += 1
        if r_err < 1 and t_err < 0.01:  # 1cm/1deg
            pct1 += 1

    total_frames = len(rErrs)
    assert total_frames == len(testset)

    # Compute median errors.
    tErrs.sort()
    rErrs.sort()
    median_idx = total_frames // 2
    median_rErr = rErrs[median_idx]
    median_tErr = tErrs[median_idx]

    # Compute final metrics.
    pct10_5 = pct10_5 / total_frames * 100
    pct5 = pct5 / total_frames * 100
    pct2 = pct2 / total_frames * 100
    pct1 = pct1 / total_frames * 100

    _logger.info("===================================================")
    _logger.info("Test complete.")

    _logger.info('Accuracy:')
    _logger.info(f'\t10cm/5deg: {pct10_5:.1f}%')
    _logger.info(f'\t5cm/5deg: {pct5:.1f}%')
    _logger.info(f'\t2cm/2deg: {pct2:.1f}%')
    _logger.info(f'\t1cm/1deg: {pct1:.1f}%')

    _logger.info(f"Median Error: {median_rErr:.1f}deg, {median_tErr:.1f}cm")
