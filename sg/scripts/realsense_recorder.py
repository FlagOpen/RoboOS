import pyrealsense2 as rs
import numpy as np
from PIL import Image
from sg.utils.utils import WH, RECORDER_DIR, DEPTH_MIN, DEPTH_MAX
import os
import shutil
import cv2
import time
import threading
from IPython import embed
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Union

# The actual image size required to ensure that 
# the depth and color closer to the camera are not captured
#real_height = 600  # remove bottom 200px to remove robot arm in image bottom
#real_width = 1200
#real_height = 768
#real_width = 1024
real_height = 480
real_width = 640

class RecorderImage():
    def __init__(self, recorder_dir=None, serial_number="239222303321", 
                 WH=WH, FPS=30, depth_threshold=[DEPTH_MIN, DEPTH_MAX]):
        
        if recorder_dir is not None:
            self.record_flag = True
            self.recorder_dir = recorder_dir
            self.transforms = {}

            if os.path.isdir(self.recorder_dir):
                if input("Overwrite file y/n?: ") != "n":
                    shutil.rmtree(self.recorder_dir)
                else:
                    return
            os.makedirs(self.recorder_dir / "depth", exist_ok=True)
            os.makedirs(self.recorder_dir / "rgb", exist_ok=True)
            os.makedirs(self.recorder_dir / "point", exist_ok=True)
            os.makedirs(self.recorder_dir / "mask", exist_ok=True)
        
        self.WH = WH
        self.serial_number = serial_number
        self.FPS = FPS
        self.depth_threshold = depth_threshold

        self.config = rs.config()
        # Specify the wrist camera serial number
        self.config.enable_device(self.serial_number)
        self.pipeline = rs.pipeline()
        self.config.enable_stream(rs.stream.depth, self.WH[0], self.WH[1], rs.format.z16, self.FPS)
        self.config.enable_stream(rs.stream.color, self.WH[0], self.WH[1], rs.format.bgr8, self.FPS)
        profile = self.pipeline.start(self.config)
        # Skip 15 first frames to give the Auto-Exposure time to adjust
        for x in range(15):
            self.pipeline.wait_for_frames()
        color_stream = profile.get_stream(rs.stream.color)
        self.intrinsic = color_stream.as_video_stream_profile().get_intrinsics()

        self.intrinsic_matrix, self.dist_coef = self._get_readable_intrinsic()
        print(self.intrinsic_matrix, self.dist_coef)
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        print("Depth Scale is: " , self.depth_scale)
        self.clipping_distance = 1 / self.depth_scale
        print(self.clipping_distance)
        # Initialize depth process
        self._init_depth_process()
        self.frame_index = 0
        self.data = {}

    def _init_depth_process(self):
        # Initialize the processing steps
        self.depth_to_disparity = rs.disparity_transform(True)
        self.disparity_to_depth = rs.disparity_transform(False)
        self.spatial = rs.spatial_filter()
        self.spatial.set_option(rs.option.filter_magnitude, 5)
        self.spatial.set_option(rs.option.filter_smooth_alpha, 0.75)
        self.spatial.set_option(rs.option.filter_smooth_delta, 1)
        self.spatial.set_option(rs.option.holes_fill, 1)
        self.temporal = rs.temporal_filter()
        self.temporal.set_option(rs.option.filter_smooth_alpha, 0.75)
        self.temporal.set_option(rs.option.filter_smooth_delta, 1)
        # Initialize the alignment to make the depth data aligned to the rgb camera coordinate
        self.align = rs.align(rs.stream.color)

    def _get_readable_intrinsic(self):
        intrinsic_matrix = np.array(
            [
                [self.intrinsic.fx, 0, self.intrinsic.ppx],
                [0, self.intrinsic.fy, self.intrinsic.ppy],
                [0, 0, 1],
            ]
        )
        dist_coef = np.array(self.intrinsic.coeffs)
        return intrinsic_matrix, dist_coef


    def project_point_to_pixel(self, points):
        # The points here should be in the camera coordinate, n*3
        points = np.array(points)
        pixels = []
        # # Use the realsense projeciton, however, it's slow for the loop; this can give nan for invalid points
        # for i in range(len(points)):
        #     pixels.append(rs.rs2_project_point_to_pixel(self.intrinsic, points))
        # pixels = np.array(pixels)

        # Use the opencv projection
        # The width and height are inversed here
        pixels = cv2.projectPoints(
            points,
            np.zeros(3),
            np.zeros(3),
            self.intrinsic_matrix,
            self.dist_coef,
        )[0][:, 0, :]

        return pixels[:, ::-1]

    def deproject_pixel_to_point(self, pixel_depth):
        # pixel_depth contains [i, j, depth[i, j]]
        points = []
        for i in range(len(pixel_depth)):
            # The width and height are inversed here
            points.append(
                rs.rs2_deproject_pixel_to_point(
                    self.intrinsic,
                    [pixel_depth[i, 1], pixel_depth[i, 0]],
                    pixel_depth[i, 2],
                )
            )
        return np.array(points)

    def _process_depth(self, depth_frame):
        # Depth process
        filtered_depth = self.depth_to_disparity.process(depth_frame)
        filtered_depth = self.spatial.process(filtered_depth)
        filtered_depth = self.temporal.process(filtered_depth)
        filtered_depth = self.disparity_to_depth.process(filtered_depth)
        return filtered_depth

    def get_observations(self):
        width, height = self.WH
        depth_frame = None
        for x in range(5):  # each frame past five frame
            self.pipeline.wait_for_frames()
        while not depth_frame:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

        # Depth process
        filtered_depth = self._process_depth(depth_frame)
        # Calculate the pointcloud
        pointcloud = rs.pointcloud()
        pointcloud.map_to(color_frame)
        pointcloud = pointcloud.calculate(filtered_depth)
        # Get the 3D points and colors
        points = (
            np.asanyarray(pointcloud.get_vertices())
            .view(np.float32)
            .reshape([height, width, 3])
        )
        # Convert the colors from BGR to RGB
        # colors = (np.asanyarray(color_frame.get_data()) / 255.0)[:, :, ::-1]
        # # Get the depth image, make the depth in meters
        # depths = np.asanyarray(filtered_depth.get_data()) / self.clipping_distance
        colors = np.asanyarray(color_frame.get_data())
        # Get the depth image, make the depth in meters
        depths = np.asanyarray(filtered_depth.get_data())
        # Get the mask of the valid depth in the depth threshold
        mask = np.logical_and(
            (depths > self.depth_threshold[0] * self.clipping_distance), 
            (depths < self.depth_threshold[1] * self.clipping_distance)
        )
        return points, colors, depths, mask


    def get_align_frame(self, frame_index):
        # When getting_frame, you cannot crop directly, otherwise you may not get a valid frame.
        try:
            points, colors, depths, mask = self.get_observations()
            color_image_path = self.recorder_dir / "rgb" / f"{frame_index:06}.jpg"
            depth_path = self.recorder_dir / "depth" / f"{frame_index:06}.npy"
            point_path = self.recorder_dir / "point" / f"{frame_index:06}.npy"
            mask_path = self.recorder_dir / "mask" / f"{frame_index:06}.npy"
            cv2.imwrite(str(color_image_path), colors, [cv2.IMWRITE_JPEG_QUALITY, 100])
            np.save(str(depth_path), depths)
            np.save(str(point_path), points)
            np.save(str(mask_path), mask)
        except:
            return False
        return True

    def change_size(self):
        os.makedirs(self.recorder_dir / "calibration", exist_ok=True)
        # change depth and color's shape to 384 * 512
        depth_image_dir = self.recorder_dir / "depth"
        color_image_dir = self.recorder_dir / "rgb"
        point_dir = self.recorder_dir / "point"
        mask_dir = self.recorder_dir / "mask"
        depth_paths = [f for f in sorted(depth_image_dir.iterdir(), key=lambda x: int(x.stem))]
        color_paths = [f for f in sorted(color_image_dir.iterdir(), key=lambda x: int(x.stem))]
        point_paths = [f for f in sorted(point_dir.iterdir(), key=lambda x: int(x.stem))]
        mask_paths = [f for f in sorted(mask_dir.iterdir(), key=lambda x: int(x.stem))]

        self.length = len(depth_paths)
        for i in tqdm(range(self.length), desc="change size: "):
            # Image.fromarray(np.asarray(Image.open(depth_paths[i]), dtype=np.uint16)[:384, 64:576]).save(depth_paths[i])
            # Image.fromarray(np.asarray(Image.open(color_paths[i]), dtype=np.uint8)[:384, 64:576, :]).save(color_paths[i])
            cv2.imwrite(str(color_paths[i]), cv2.imread(str(color_paths[i]))[:600, :, :])
            np.save(depth_paths[i], np.load(depth_paths[i])[:600, :])
            np.save(point_paths[i], np.load(point_paths[i])[:600, :])
            np.save(mask_paths[i], np.load(mask_paths[i])[:600, :])
            calibration_path = self.recorder_dir / "calibration" / f"{i:06}.txt"
            np.savetxt(str(calibration_path), self.intrinsic_matrix)

    def set_metadata(self):
        metadata = {
            "w": real_width,
            "h": real_height, 
            "dw": real_width, 
            "dh": real_height,
            "fps": self.FPS,
            "K": self.intrinsic_matrix.tolist(),
            "depth_scale": self.depth_scale,
            "min_depth": DEPTH_MIN,
            "max_depth": DEPTH_MAX,
            "cameraType": 1,
            "dist_coef": self.dist_coef.tolist(),
            "length": self.length
        }
        with open(self.recorder_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

    def start_record(self):
        while self.record_flag:
            flag = self.get_align_frame(self.frame_index)
            # time.sleep(0.001)
            # time.sleep(0.1)
            # print(self.frame_index)
            if flag: self.frame_index += 1
    
    def stop_record(self):
        self.record_flag = False
        self.pipeline.stop()


def record():
    if input("Do you want to record data? [y/n]: ") == "n":
        return 
    recorder_dir = RECORDER_DIR / "test"
    imagerecorder = RecorderImage(recorder_dir=recorder_dir)
    input("\033[32mPress any key to Start.\033[0m")
    record_thread = threading.Thread(target=imagerecorder.start_record)
    record_thread.start()
    input("\033[31mRecording started. Press any key to stop.\033[0m")
    imagerecorder.stop_record()
    record_thread.join()
    imagerecorder.change_size()
    imagerecorder.set_metadata()
    intrinsic = imagerecorder.intrinsic
    with open(recorder_dir / "calib.txt", "w") as f:
        f.write(f'{intrinsic.fx} {intrinsic.fy} {intrinsic.ppx} {intrinsic.ppy}')
    # imagerecorder.depth_to_color_vis()
    print(f"All Images are save in {imagerecorder.recorder_dir}: depth / rgb, length is {len(os.listdir(imagerecorder.recorder_dir / 'depth'))}")

    del imagerecorder



if __name__ == "__main__":
    record()
