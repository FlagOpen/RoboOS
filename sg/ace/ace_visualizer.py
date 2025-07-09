# Copyright © Niantic, Inc. 2022.

import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import logging
import math
import numpy as np
import pyrender
from skimage.transform import rotate
from skimage import io, draw
import matplotlib.pyplot as plt
import pickle

import ace.ace_vis_util as vutil

_logger = logging.getLogger(__name__)

class ACEVisualizer:
    """
    Creates a representation of a scene (camera poses, scene point cloud) and write video frames.

    Supports mapping phase and relocalisation phase, and a smooth transition inbetween. For the mapping phase,
    the visualiser shows the mapping camera trajectory and the training process of the scene point cloud. For the
    relocalisation phase, the visualiser follows the query camera, and keeps a visual trace of relocalised poses.

    The visualiser has an internal state of three buffers that comprise all aspects of the next frame to be rendered.
    A call to _render_frame_from_buffers() will generate an image from these buffers.
    The three buffers are:

        self.scene_camera: To get the current rendering view.
        self.trajectory_buffer: Mesh geometry that represents camera frustums and trajectories.
        self.point_cloud_buffer: Point cloud geometry that represents the scene model.

    The class contains many helper classes to manipulate and fill these buffers. There are also function to manipulate
    the rendered frame before storing it, e.g. to add error histograms.

    The main interface for the mapping stage:
    1) setup_mapping_visualisation: Called once in the beginning, resets buffers, creates mapping camera trajectory
    2) render_mapping_frame: Called during learning iterations, shows currents snapshot of the scene point cloud
    3) finalize_mapping: Renders several frames that show the fully trained point cloud, stores buffers on disk
        so that the relocalisation script can resume smoothly

    The main interface for the relocalisation stage:
    1) setup_reloc_visualisation: Called once in the beginning, tries to load buffers of the mapping stage from disk
    2) render_reloc_frame: Called for each query image during relocalisation
    """

    def __init__(self, target_path, flipped_portait, map_depth_filter, mapping_vis_error_threshold=10, reloc_vis_error_threshold=20):
        """
        Constructor. Sets standard values for visualisation parameters.

        @param target_path: where rendered frames will be stored
        @param flipped_portait: whether dataset images are 90 degree rotated portrait images
        @param map_depth_filter: filters scene point cloud by distance to the camera, removing far away points (meters)
        @param mapping_vis_error_threshold: threshold when mapping the reprojection error to a color map (pixels)
        @param reloc_vis_error_threshold: threshold when mapping the pose error to a color map (cm/deg)
        """
        self.target_path = target_path
        # buffer file for smooth rendering across training and test script calls
        self.state_file = os.path.join(self.target_path, 'mapping_state.pkl')

        # flip rendering by 90deg if dataset is stored as flipped portrait images (e.g. Wayspots)
        self.flipped_portrait = flipped_portait

        self.map_depth_filter = map_depth_filter

        # main visualisation parameters
        self.render_width  = 1280  # output image resolution
        self.render_height = 720 # output image resolution
        self.point_size = 2.0

        if self.flipped_portrait:
            # for flipped portrait datasets, we render sideways and rotate the final image back
            self.render_width, self.render_height = self.render_height, self.render_width

        reference_height = min(self.render_height, self.render_width)
        self.err_hist_bins = 20
        self.err_hist_x = int(0.05 * reference_height)
        self.err_hist_y = int(1.35 * reference_height)
        self.err_hist_h = int(0.4 * reference_height)
        self.err_hist_w_reloc = int(0.6 * reference_height)
        self.err_hist_w_mapping = int(0.2 * reference_height)

        # mapping vis parameters
        self.framecount_transition = 40 # frame count for growing the fully trained map at the end of mapping
        self.pan_angle_coverage = 60 # degrees, opening angle of the camera pan
        self.frustum_scale_mapping = 0.3 # scale factor for the camera frustum objects
        self.mapping_frustum_skip = 0.5 # place mapping frustum every X meters

        self.mapping_vis_error_threshold = mapping_vis_error_threshold # threshold on reprojection error in px (for color mapping)
        self.mapping_color_map = vutil.get_retro_colors() # dark magenta to bright cyan

        # reloc vis parameters
        self.frustum_scale_reloc = 1.0  # scale factor for the camera frustum objects
        self.reloc_vis_error_threshold = reloc_vis_error_threshold  # threshold on pose error in cm/deg (for color mapping)
        self.reloc_color_map = plt.cm.get_cmap("summer")(np.linspace(0, 1, 256))[1:, :3] #green to yellow
        self.reloc_color_outlier = (200, 0, 0) # red

        # remember last frame's estimate and error color to add a marker to the camera trajectory
        self.reloc_buffer_previous_est = None
        self.reloc_buffer_previous_clr = None
        self.reloc_frame_count = 0
        # remember all reloc errors so far
        self.reloc_error_buffer = None
        # option to skip rendering reloc frames for dense and long sequences (like 7Scenes)
        self.reloc_frame_skip = 0
        self.reloc_frame_counter = 0

        # camera views for rendering the scene during mapping
        self.pan_cams = None
        self.mapping_frame_count = 0

        # buffer for observing camera
        self.scene_camera = None

        # camera trajectory to render
        self.trajectory_buffer = None

        # buffer holding the map point cloud
        self.point_cloud_buffer = None

        # index of current frame, rendered frame
        self.frame_idx = 0

    def _generate_camera_pan(self,
                             pan_number_cams,
                             mapping_poses,
                             pan_angle_coverage):
        """
        Generate a list of camera views that smoothly pan around the scene.

        @param pan_number_cams: Number of views to be generated.
        @param mapping_poses: Mapping camera poses that the pan should enclose.
        @param pan_angle_coverage: Opening angle of the pan (degrees).
        @return: List of 4x4 camera poses.
        """
        pan_cams = []

        # select anchor camera to be used for the mapping camera pan
        pan_center_pose = mapping_poses[len(mapping_poses) // 2].copy()

        # move pan center to the average of all pose positions
        poses_pos = [pose[:3, 3] for pose in mapping_poses]
        poses_pos = np.stack(poses_pos, axis=-1)
        pan_center_pose[:3, 3] = poses_pos.mean(axis=1)

        # get approximate extent of mapping cameras
        poses_pos_extent = poses_pos.max(axis=1) - poses_pos.min(axis=1)
        poses_extent = [poses_pos_extent[c] for c in range(3)]
        # hack to support different coordinate conventions
        # find the two axis of maximum extent and average those
        poses_extent.sort(reverse=True)
        poses_extent = 0.5 * (poses_extent[0] + poses_extent[1])

        # create a camera pan around the scene
        pan_radius = 0.5 * poses_extent

        pan_angle_start = -90 - pan_angle_coverage / 2
        pan_angle_increment = pan_angle_coverage / pan_number_cams

        for i in range(pan_number_cams):
            pan_pose = np.eye(4)

            pan_angle = math.radians(pan_angle_start + pan_angle_increment * i)
            pan_x = pan_radius * math.cos(pan_angle)
            pan_z = -pan_radius * math.sin(pan_angle)

            x_axis_index = 0
            if self.flipped_portrait:
                x_axis_index = 1

            pan_pose[x_axis_index, 3] = pan_x
            pan_pose[2, 3] = pan_z

            if self.flipped_portrait:
                # rotation around x
                pan_rotation_angle = math.radians(pan_angle_coverage / 2 - pan_angle_increment * i)

                pan_pose[1, 1] = math.cos(pan_rotation_angle)
                pan_pose[1, 2] = -math.sin(pan_rotation_angle)
                pan_pose[2, 1] = math.sin(pan_rotation_angle)
                pan_pose[2, 2] = math.cos(pan_rotation_angle)
            else:
                # rotation around y
                pan_rotation_angle = math.radians(-pan_angle_coverage / 2 + pan_angle_increment * i)

                pan_pose[0, 0] = math.cos(pan_rotation_angle)
                pan_pose[0, 2] = math.sin(pan_rotation_angle)
                pan_pose[2, 0] = -math.sin(pan_rotation_angle)
                pan_pose[2, 2] = math.cos(pan_rotation_angle)

            pan_pose = pan_center_pose @ pan_pose

            pan_cams.append(pan_pose)

        return pan_cams

    def _generate_camera_trajectory(self,
                                    mapping_poses,
                                    mapping_image_paths):
        """
        Add all mapping cameras to the trajectory buffer.

        @param mapping_poses: List of camera poses (4x4)
        @param mapping_image_paths: Corresponding list of image files.
        """
        for frame_idx in range(len(mapping_poses)):

            # get pose of mapping camera
            frustum_pose = mapping_poses[frame_idx].copy()

            # add new camera to trajectory line
            self.trajectory_buffer.grow_camera_path(frustum_pose)

            # add camera frustum to trajectory
            self.trajectory_buffer.add_camera_frustum(frustum_pose,
                                                      image_file=mapping_image_paths[frame_idx],
                                                      sparse=True)

    @staticmethod
    def _convert_cv_to_gl(pose):
        """
        Convert a pose from OpenCV to OpenGL convention (and vice versa).

        @param pose: 4x4 camera pose.
        @return: 4x4 camera pose.
        """
        gl_to_cv = np.array([[1, -1, -1, 1], [-1, 1, 1, -1], [-1, 1, 1, -1], [1, 1, 1, 1]])
        return gl_to_cv * pose

    def setup_mapping_visualisation(self,
                                    pose_files,
                                    image_files,
                                    frame_count,
                                    camera_z_offset
                                    ):
        """
        Reset visualisation buffers for the mapping visualisation.

        Generate mapping camera pan, and create the mapping trajectory mesh.

        @param pose_files: List of mapping pose files, assumed to contain 4x4 matrices in OpenCV convention.
        @param image_files: Corresponding list of mapping image files.
        @param frame_count: How many frames we plan to render for the mapping stage.
        @param camera_z_offset: Distance from the rendering camera (meters), can be used to zoom in/out depending on scene size.
        """
        _logger.info("Setting up mapping visualisation.")

        # make sure poses and image files are in sync (relying on the file naming schema)
        pose_files.sort()
        image_files.sort()

        # load mapping poses (4x4 matrices, camera to scene coordinates, OpenGL convention)
        mapping_poses = [self._convert_cv_to_gl(np.loadtxt(pose_file)) for pose_file in pose_files]

        # create panning motion around scene
        self.pan_cams = self._generate_camera_pan(
            pan_number_cams=frame_count+self.framecount_transition,
            mapping_poses=mapping_poses,
            pan_angle_coverage=self.pan_angle_coverage
        )

        # reset camera trajectory to render
        self.trajectory_buffer = vutil.CameraTrajectoryBuffer(
            frustum_skip=self.mapping_frustum_skip,
            frustum_scale=self.frustum_scale_mapping
        )
        # fill buffer with mapping trajectory
        self._generate_camera_trajectory(mapping_poses, image_files)

        # reset frame counter
        self.frame_idx = 0
        self.mapping_frame_count = frame_count

        # reset scene camera
        self.scene_camera = vutil.LazyCamera(backwards_offset=camera_z_offset)

        # reset mapping point cloud buffer
        self.point_cloud_buffer = vutil.PointCloudBuffer()

    @staticmethod
    def _render_pc(r, pc, camera, camera_pose):
        """
        Render a point cloud on a black background.

        @param r: PyRender Renderer.
        @param pc: PyRender point cloud object.
        @param camera: PyRender camera object.
        @param camera_pose: 4x4 camera pose.
        @return: Rendered frame (RGB).
        """
        scene = pyrender.Scene(bg_color=(0, 0, 0), ambient_light=(1, 1, 1))
        scene.add(pc)
        scene.add(camera, pose=camera_pose)
        color, _ = r.render(scene)

        return color

    @staticmethod
    def _render_trajectory(r, trajectory, camera, camera_pose, frustum_images):
        """
        Renders the trajectory mesh with flat lighting on a transparent background.

        @param r: PyRender Renderer.
        @param trajectory: PyRender mesh object.
        @param camera: PyRender camera object.
        @param camera_pose: 4x4 camera pose.
        @param frustum_images: Textured meshes that represent image boxes.
        @return: Rendered frame (RGBA).
        """
        scene = pyrender.Scene(bg_color=(0, 0, 0, 0), ambient_light=(1, 1, 1))
        scene.add(trajectory)
        scene.add(camera, pose=camera_pose)

        for frustum_image in frustum_images:
            scene.add(frustum_image)

        color, _ = r.render(scene, flags=(pyrender.constants.RenderFlags.RGBA | pyrender.constants.RenderFlags.FLAT))

        return color

    @staticmethod
    def _blend_images(img1_RGB, img2_RGBA):
        """
        Add an RGBA image on top of an RGB image.

        @param img1_RGB: Background image.
        @param img2_RGBA: Transparent image for blending on top.
        @return: Blended image (RGB)
        """
        mask = img2_RGBA[:, :, 3].astype(float)
        mask /= 255
        mask = np.expand_dims(mask, axis=2)

        blended_rgb = img2_RGBA[:, :, :3].astype(float) * mask + img1_RGB.astype(float) * (1 - mask)
        return blended_rgb.astype('uint8')

    def _errors_to_colors(self, errors, max_error):
        """
        Map errors to error color map (self.mapping_color_map).

        @param errors: 1D array of N scalar errors
        @param max_error: Error threshold for mapping to the color map
        @return: Color array N3 and normalized error array N1
        """
        # map reprojection error up to X pixels to color map
        norm_errors = errors / max_error  # normalise
        norm_errors = 1 - norm_errors.clip(0, 1)  # reverse

        # error indices for color map
        errors_idxs = (norm_errors * 255).astype(int)

        # expand color map to size of the point cloud
        errors_clr = np.broadcast_to(self.mapping_color_map, (errors_idxs.shape[0], 256, 3))
        # for each point, pick color from color map according to error index
        errors_clr = errors_clr[np.arange(errors_idxs.shape[0]), errors_idxs] * 255

        return errors_clr, norm_errors

    def _get_mapping_progress(self):
        """
        Get percentage of mapping done.

        @return: Scalar (0,1)
        """
        return min(self.frame_idx / self.mapping_frame_count, 1.0)

    def _draw_loading_bar(self, image):
        """
        Draw a 2D loading bar with the current percentage of mapping done to the image.

        @param image: Input frame.
        @return: Frame with loading bar.
        """
        image_h = image.shape[0]

        loading_bar_x = int(0.93 * image_h)
        loading_bar_y = int(0.27 * image_h)
        loading_bar_h = int(0.04 * image_h)
        loading_bar_w = 1.215 * image_h

        loading_bar_start = (loading_bar_x, loading_bar_y)
        loading_bar_progress = self._get_mapping_progress()
        loading_bar_extent = (loading_bar_h, int(loading_bar_progress * loading_bar_w))

        rr, cc = draw.rectangle(loading_bar_start, extent=loading_bar_extent)
        image[rr, cc, 0:3] = 0.8 * image[rr, cc, 0:3] + 0.2 * 255

        loading_bar_extent = (int(0.04 * image_h), int(1.215 * image_h))
        rr, cc = draw.rectangle_perimeter(loading_bar_start, extent=loading_bar_extent)
        image[rr, cc, 0:3] = 255

        return image

    def _draw_hist(self, image, hist_values, hist_colors, hist_x, hist_y, hist_w, hist_h, hist_max):
        """
        Add a histogram to the frame.

        @param image: Input frame.
        @param hist_values: Values of histogram bars.
        @param hist_colors: RGB color for each bar.
        @param hist_x: Horizontal position in pixels.
        @param hist_y: Vertical position in pixels.
        @param hist_w: Width in pixels.
        @param hist_h: Height in pixels.
        @param hist_max: Normalising factor for hist_values.
        """
        hist_bins = len(hist_values)
        bar_h = int(hist_h / hist_bins)

        for hist_idx in range(hist_bins):
            bar_w = int(hist_w * (hist_values[hist_idx] / hist_max))
            bar_w = max(3, bar_w)
            bar_y = int(hist_y + hist_idx * bar_h)

            # draw the actual colored bars
            rr, cc = draw.rectangle((hist_x, bar_y), extent=(bar_w, bar_h))
            image[rr, cc, 0:3] = hist_colors[hist_idx]

    def _draw_repro_error_hist(self, image, errors):
        """
        Draw histogram of mapping reprojection errors.

        @param image: Input frame.
        @param errors: 1D array of scalar reprojection errors.
        @return: Frame with histogram.
        """
        # generate histogram of reprojection errors (normalized between 0 and 1 already)
        hist_values, _ = np.histogram(errors, bins=self.err_hist_bins, range=(0, 1))

        # look up colors for bins
        hist_color_idxs = [int(hist_idx / self.err_hist_bins * 255) for hist_idx in range(self.err_hist_bins)]
        hist_colors = [self.mapping_color_map[clr_idx] * 255 for clr_idx in hist_color_idxs]

        self._draw_hist(
            image=image,
            hist_values=hist_values,
            hist_colors=hist_colors,
            hist_x=self.err_hist_x,
            hist_y=self.err_hist_y,
            hist_h=self.err_hist_h,
            hist_w=self.err_hist_w_mapping,
            hist_max=hist_values.max())

        return image

    def _draw_pose_error_hist(self, image, errors):
        """
        Draw histogram of relocalisation pose errors.

        @param image: Input frame.
        @param errors: 1D array of scalar pose errors.
        @return: Frame with histogram.
        """

        # generate histogram of pose errors, with last bin collecting all outlier estimates
        hist_values = np.zeros((self.err_hist_bins,))
        # handle n-1 bins as usual
        hist_values[:self.err_hist_bins - 1], _ = np.histogram(errors,
                                                               bins=self.err_hist_bins - 1,
                                                               range=(0, self.reloc_vis_error_threshold))
        # manually create fill last outlier bin
        hist_values[self.err_hist_bins - 1] = sum(error > self.reloc_vis_error_threshold for error in errors)

        # look up colors for bins but special handling of outlier bin
        hist_color_idxs = [int(hist_idx / self.err_hist_bins * 255) for hist_idx in range(self.err_hist_bins-1)]
        hist_colors = [self.reloc_color_map[clr_idx] * 255 for clr_idx in hist_color_idxs]
        hist_colors.append(self.reloc_color_outlier) # special handling of last bin

        self._draw_hist(
            image=image,
            hist_values=hist_values,
            hist_colors=hist_colors,
            hist_x=self.err_hist_x,
            hist_y=self.err_hist_y,
            hist_h=self.err_hist_h,
            hist_w=self.err_hist_w_reloc,
            hist_max=self.reloc_frame_count)

        return image

    @staticmethod
    def _write_captions(image, captions_dict, text_color=(1, 1, 1)):
        """
        Write text onto frame.

        Using matplotlib following https://scikit-image.org/docs/stable/auto_examples/applications/plot_text.html

        @param image: Input frame.
        @param captions_dict: Dictionary specifying multiple captions, with fields x, y, text and fs (font size).
        @param text_color: RGB color of text.
        @return: Frame with text.
        """
        fig = plt.figure()
        fig.figimage(image, resize=True)

        for caption in captions_dict:
            fig.text(caption['x'], caption['y'], caption['text'], fontsize=caption['fs'], va="top", color=text_color)

        fig.canvas.draw()
        image = np.asarray(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)

        return image

    def _write_mapping_captions(self, image):
        """
        Write all image captions for the mapping stage.

        @param image: Input frame.
        @return: Frame with captions.
        """
        image_h = image.shape[0]

        captions_dict = [
            {'x': 0.15, 'y': 0.25, 'fs': 0.07 * image_h,
             'text': "5 Minute Neural Mapping"},
            {'x': 0.15, 'y': 0.15, 'fs': 0.04005 * image_h,
             'text': "Learning the scene from posed RGB images"},
            {'x': 0.47, 'y': 0.063, 'fs': 0.02 * image_h,
             'text': f"{self._get_mapping_progress() * 100:000.1f}%"},
            {'x': 0.76, 'y': 0.975, 'fs': 0.015 * image_h,
             'text': f">{self.mapping_vis_error_threshold}px       Reprojection Error       0px"}
            # yep, I use spaces to align parts of this caption. Too lazy to do it properly :)
        ]

        return self._write_captions(image, captions_dict)

    def _write_reloc_captions(self, image):
        """
        Write all image captions for the relocalisation stage.

        @param image: Input frame.
        @return: Frame with captions.
        """
        image_h = image.shape[0]

        captions_dict = [
            {'x': 0.32, 'y': 0.25, 'fs': 0.07 * image_h,
             'text': "Relocalisation"},
            {'x': 0.11, 'y': 0.15, 'fs': 0.04005 * image_h,
             'text': "Estimating camera pose from a single RGB image"},
            {'x': 0.76, 'y': 0.975, 'fs': 0.015 * image_h,
             'text': f"0cm      Camera Pose Error      >{self.reloc_vis_error_threshold}cm"}
            # yep, I use spaces to align parts of this caption. Too lazy to do it properly :)
        ]

        return self._write_captions(image, captions_dict)

    def _render_frame_from_buffers_safe(self):
        """
        Wrapper for _render_frame_from_buffers, re-trying rendering if render lib throws error.

        We found the rendering backend to be brittle, throwing random errors now and then.
        Re-trying to render the same geometry worked always.

        @return: rendered frame or None if rendering failed after multiple tries
        """
        max_tries = 10

        while max_tries > 0:
            try:
                return self._render_frame_from_buffers()
            except:
                _logger.warning("Rendering failed, trying again!")
                max_tries -= 1

        raise RuntimeError("Re-rendering failed too often...")

    def _render_frame_from_buffers(self):
        """
        Render current frame according to state of internal buffers: scene camera, point cloud and trajectory mesh.

        @return: Rendered frame.
        """
        # get smooth observing camera
        smooth_camera_pose = self.scene_camera.get_current_view()

        # initialise pyrender pipeline
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=self.render_width / self.render_height)
        r = pyrender.OffscreenRenderer(self.render_width, self.render_height, point_size=self.point_size)

        # cast PC to rendering object
        frame_xyz, frame_clr, _ = self.point_cloud_buffer.get_point_cloud()
        ace_map = pyrender.Mesh.from_points(frame_xyz, colors=frame_clr)

        # get camera trajectory mesh
        trajectory_mesh, frustum_images = self.trajectory_buffer.get_mesh()

        # render PC with normal shading
        bg_RGB = self._render_pc(r, ace_map, camera, smooth_camera_pose)
        # render camera trajectory with flat shading and alpha transparency for blending
        cams_RGBA = self._render_trajectory(r, trajectory_mesh, camera, smooth_camera_pose, frustum_images)
        # combine the two renders
        blended_RGB = self._blend_images(bg_RGB, cams_RGBA)

        # rotate from portrait to landscape
        if self.flipped_portrait:
            blended_RGB = rotate(blended_RGB, -90, resize=True, preserve_range=True).astype('uint8')

        return blended_RGB

    def _save_frame(self, frame):
        """
        Store frame with current frame number to target folder.

        @param frame: Input image.
        """
        out_render_file = f"{self.target_path}/frame_{self.frame_idx:05d}.png"
        io.imsave(out_render_file, frame)
        _logger.info(f"Rendered and saved frame: {out_render_file}")

    def _render_mapping_frame_from_buffers(self):
        """
        Render current frame according to buffers, and draw mapping specific captions and the reprojection error histogram.
        """
        # update observing camera
        self.scene_camera.update_camera(self.pan_cams[self.frame_idx])

        current_frame = self._render_frame_from_buffers_safe()

        if current_frame is not None:
            # draw loading bar, captions and reprojection error histogram
            _, _, frame_errs = self.point_cloud_buffer.get_point_cloud()
            current_frame = self._draw_loading_bar(current_frame)
            current_frame = self._draw_repro_error_hist(current_frame, frame_errs)
            current_frame = self._write_mapping_captions(current_frame)

            # write to disk
            self._save_frame(current_frame)

        # move frame index pointer for next render call
        self.frame_idx += 1

    def render_mapping_frame(self, scene_coordinates, errors):
        """
        Update point cloud buffer with current scene coordinates and render frame.

        Stores rendered frame to target folder.

        @param scene_coordinates: N3 array of points in OpenCV convention.
        @param errors: N1 array of scalar reprojection errors for coloring the point cloud.
        """
        # OpenCV to OpenGL convention
        scene_coordinates[:, 1] = -scene_coordinates[:, 1]
        scene_coordinates[:, 2] = -scene_coordinates[:, 2]
        # color point cloud according to errors
        scene_coordinates_clr, errors_normalized = self._errors_to_colors(errors, self.mapping_vis_error_threshold)
        # update rolling buffer
        self.point_cloud_buffer.update_buffer(scene_coordinates, scene_coordinates_clr, errors_normalized)

        # render actual frame
        self._render_mapping_frame_from_buffers()

    def finalize_mapping(self, network, data_loader):
        """
        Render final mapping frames that show the fully trained point cloud.

        Stores rendered frames to target folder.
        Stores final mapping buffers to disk, so that the relocalisation script can resume smoothly.

        @param network: Fully trained network.
        @param data_loader: Data loader for the mapping sequence, to extract point cloud with the network.
        """
        _logger.info(f"Extract fully trained map from network.")
        map_xyz, map_clr = vutil.get_point_cloud_from_network(network, data_loader, self.map_depth_filter)

        # split the full point cloud into chunks for a "growing" effect
        main_pc_chunk_size = map_xyz.shape[0] // self.framecount_transition
        main_pc_xyz_buffer = []
        main_pc_clr_buffer = []

        for i in range(self.framecount_transition):
            chunk_start = i * main_pc_chunk_size
            chunk_end = (i + 1) * main_pc_chunk_size
            main_pc_xyz_buffer.append(map_xyz[chunk_start:chunk_end])
            main_pc_clr_buffer.append(map_clr[chunk_start:chunk_end])

        _logger.info(f"Rendering final frames of map growing.")

        for transition_idx in range(self.framecount_transition):

            # update rolling buffer
            self.point_cloud_buffer.update_buffer(
                main_pc_xyz_buffer[transition_idx],
                main_pc_clr_buffer[transition_idx])

            if transition_idx == self.point_cloud_buffer.pc_buffer_size:
                # point cloud buffer has been entirely filled with new map pc
                # disable rolling buffer and just continue accumulate PC chunks for growing the full map
                self.point_cloud_buffer.disable_buffer_cap()

            # render actual frame
            self._render_mapping_frame_from_buffers()

        # save state for smooth transition when rendering the localisation phase
        state_dict = {
            'map_xyz': map_xyz,
            'map_clr': map_clr,
            'frame_idx': self.frame_idx,
            'camera_buffer': self.scene_camera.get_camera_buffer()
        }

        with open(self.state_file, "wb") as file:
            pickle.dump(state_dict, file)
        _logger.info(f"Stored rendering buffer to {self.state_file}.")

    def setup_reloc_visualisation(self, frame_count, network, data_loader, camera_z_offset, reloc_frame_skip):
        """
        Initialise buffers for the relocalisation visualisation.

        Tries to load the mapping buffers from disk for a smooth transition. If unavailable, extracts point cloud
        again from network and mapping data loader.

        @param frame_count: How many frames we are about to render, needed for the pose error histogram.
        @param network: Fully trained network in case we have to re-generate the point cloud.
        @param data_loader: Data loader for the mapping sequence in case we have to re-generate the point cloud.
        @param camera_z_offset: Distance from the query camera view (meters), used to zoom out of the scene.
        @param reloc_frame_skip: For very high query frame rates we can render only every n'th frame.
        """
        _logger.info("Setting up relocalisation visualisation.")

        try:
            with open(self.state_file, "rb") as file:
                state_dict = pickle.load(file)

            map_xyz = state_dict['map_xyz']
            map_clr = state_dict['map_clr']

            self.frame_idx = state_dict['frame_idx']
            self.scene_camera = vutil.LazyCamera(backwards_offset=camera_z_offset,
                                                 camera_buffer=state_dict['camera_buffer'])
        except:
            _logger.warning("No mapping state buffer found. Need to recreate map from network.")

            self.frame_idx = 0
            self.scene_camera = vutil.LazyCamera(backwards_offset=camera_z_offset)

            _logger.info(f"Extract fully trained map from network.")
            map_xyz, map_clr = vutil.get_point_cloud_from_network(network, data_loader, self.map_depth_filter)

        self.point_cloud_buffer = vutil.PointCloudBuffer()
        self.point_cloud_buffer.update_buffer(map_xyz, map_clr)

        # reset all buffers
        self.trajectory_buffer = vutil.CameraTrajectoryBuffer(frustum_skip=0, frustum_scale=self.frustum_scale_reloc)
        self.reloc_error_buffer = []

        self.reloc_frame_count = frame_count
        self.reloc_frame_skip = reloc_frame_skip
        self.reloc_frame_counter = 0


    def render_reloc_frame(self, query_pose, query_file, est_pose, est_error, sparse_query):
        """
        Update query trajectory with new GT pose and estimate and render frame.

        Stores rendered frame to target folder.

        @param query_pose: (pseudo) ground truth pose, 4x4, OpenCV convention
        @param query_file: image file of query
        @param est_pose: estimated pose, 4x4, OpenCV convention
        @param est_error: scalar error of estimated pose, e.g. max of rot and trans error (cm/deg)
        @param sparse_query: flag, if true we render multiple frames per query to slow it down and make it smooth
        """
        if sparse_query:
            renders_per_query = 10
            marker_size = 0.07
        else:
            renders_per_query = 1
            marker_size = 0.02

        query_pose = self._convert_cv_to_gl(query_pose)
        est_pose = self._convert_cv_to_gl(est_pose)

        # keep track of error statistics
        self.reloc_error_buffer.append(est_error)

        # map error to color
        if est_error < self.reloc_vis_error_threshold:
            max_error_idx = int(est_error / self.reloc_vis_error_threshold * 254)
            err_color = self.reloc_color_map[max_error_idx] * 256
        else:
            err_color = self.reloc_color_outlier

        # generate query trajectory
        self.trajectory_buffer.grow_camera_path(query_pose)

        # remove previous frustums, and add just the new ones from the current frame
        self.trajectory_buffer.clear_frustums()
        self.trajectory_buffer.add_camera_frustum(query_pose, image_file=query_file, sparse=False)
        self.trajectory_buffer.add_camera_frustum(est_pose, image_file=None, sparse=False, frustum_color=err_color)

        # add previous frame's estimate as a colored marker to the trajectory
        if self.reloc_buffer_previous_est is not None:
            self.trajectory_buffer.add_position_marker(
                marker_pose=self.reloc_buffer_previous_est,
                marker_color=self.reloc_buffer_previous_clr,
                marker_extent=marker_size)

        # remember this frame's estimate for next render call
        self.reloc_buffer_previous_est = est_pose
        self.reloc_buffer_previous_clr = err_color

        # decide whether to actually render this frame
        if self.reloc_frame_counter % self.reloc_frame_skip == 0:

            # for sparse queries we render multiple frames for a smooth transition
            for render_idx in range(renders_per_query):

                # update observing camera
                self.scene_camera.update_camera(query_pose)

                # render actual frame
                current_frame = self._render_frame_from_buffers_safe()

                if current_frame is not None:
                    # finalize frame
                    current_frame = self._draw_pose_error_hist(current_frame, self.reloc_error_buffer)
                    current_frame = self._write_reloc_captions(current_frame)

                    self._save_frame(current_frame)

                # move frame index pointer for next render call
                self.frame_idx += 1

        self.reloc_frame_counter += 1


