#!/usr/bin/env python3
# Copyright © Niantic, Inc. 2022.

import logging
from argparse import ArgumentParser
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

_logger = logging.getLogger(__name__)

@dataclass
class FrameResult:
    inlier_count: int = 0
    quaternion: List[float] = field(default_factory=lambda: [1, 0, 0, 0])
    translation: List[float] = field(default_factory=lambda: [0, 0, 0])
    r_err: float = 0
    t_err: float = 0


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser(
        description="Merge results created by multiple nets trained on clustered datasets, "
                    "keeping the best pose for each image (in terms of inlier count).")
    parser.add_argument('poses_path', type=Path,
                        help="Path to a folder containing the estimated poses for each network.")
    parser.add_argument('out_file', type=Path,
                        help="Path to the output file containing the best pose for each image.")
    parser.add_argument('--poses_suffix', type=str, default='.txt', help='Suffix to select a subset of pose files.')

    args = parser.parse_args()

    poses_path: Path = args.poses_path
    out_file: Path = args.out_file

    pose_files = sorted(poses_path.glob(f"poses_*{args.poses_suffix}"))
    _logger.info(f"Found {len(pose_files)} pose files.")

    frame_poses = defaultdict(FrameResult)

    for in_file in pose_files:
        _logger.info(f"Parsing results from: {in_file}")
        with in_file.open('r') as f:
            for line in f.readlines():
                current_result = FrameResult()
                img, current_result.quaternion[0], current_result.quaternion[1], current_result.quaternion[2], \
                current_result.quaternion[3],\
                current_result.translation[0], current_result.translation[1], current_result.translation[2],\
                current_result.r_err, current_result.t_err, current_result.inlier_count = line.split()

                # Convert to the appropriate datatypes.
                current_result.inlier_count = int(current_result.inlier_count)
                current_result.quaternion = [float(x) for x in current_result.quaternion]
                current_result.translation = [float(x) for x in current_result.translation]
                current_result.r_err = float(current_result.r_err)
                current_result.t_err = float(current_result.t_err)

                # Update global dict if needed.
                if frame_poses[img].inlier_count < current_result.inlier_count:
                    frame_poses[img] = current_result

    _logger.info(f"Found results for {len(frame_poses)} query frames.")

    # Save the output.
    with out_file.open('w') as f:
        for img_name in sorted(frame_poses.keys()):
            frame_result = frame_poses[img_name]
            f.write(
                f"{img_name} "
                f"{' '.join(str(x) for x in frame_result.quaternion)} "
                f"{' '.join(str(x) for x in frame_result.translation)} "
                f"{frame_result.r_err} {frame_result.t_err} {frame_result.inlier_count}\n")
    _logger.info(f"Saved merged poses to: {out_file}")
