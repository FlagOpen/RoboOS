import logging
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from skimage import color
from skimage import io
from skimage.transform import rotate, resize
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from torchvision import transforms

from ace_network import Regressor

_logger = logging.getLogger(__name__)


class CamLocDataset_test(Dataset):
    """Camera localization dataset.

    Access to image, calibration and ground truth data given a dataset directory.
    """

    def __init__(self,
                 root_dir,
                 mode=0,
                 sparse=False,
                 augment=False,
                 aug_rotation=15,
                 aug_scale_min=2 / 3,
                 aug_scale_max=3 / 2,
                 aug_black_white=0.1,
                 aug_color=0.3,
                 image_height=480,
                 use_half=True,
                 num_clusters=None,
                 cluster_idx=None,
                 ):
        """Constructor.

        Parameters:
            root_dir: Folder of the data (training or test).
            mode:
                0 = RGB only, load no initialization targets. Default for the ACE paper.
                1 = RGB + ground truth scene coordinates, load or generate ground truth scene coordinate targets
                2 = RGB-D, load camera coordinates instead of scene coordinates
            sparse: for mode = 1 (RGB+GT SC), load sparse initialization targets when True, load dense depth maps and
                generate initialization targets when False
            augment: Use random data augmentation, note: not supported for mode = 2 (RGB-D) since pre-generated eye
                coordinates cannot be augmented
            aug_rotation: Max 2D image rotation angle, sampled uniformly around 0, both directions, degrees.
            aug_scale_min: Lower limit of image scale factor for uniform sampling
            aug_scale_min: Upper limit of image scale factor for uniform sampling
            aug_black_white: Max relative scale factor for image brightness/contrast sampling, e.g. 0.1 -> [0.9,1.1]
            aug_color: Max relative scale factor for image saturation/hue sampling, e.g. 0.1 -> [0.9,1.1]
            image_height: RGB images are rescaled to this maximum height (if augmentation is disabled, and in the range
                [aug_scale_min * image_height, aug_scale_max * image_height] otherwise).
            use_half: Enabled if training with half-precision floats.
            num_clusters: split the input frames into disjoint clusters using hierarchical clustering in order to train
                an ensemble model. Clustering is deterministic, so multiple training calls with the same number of
                target clusters will result in the same split. See the paper for details of the approach. Disabled by
                default.
            cluster_idx: If num_clusters is not None, then use this parameter to choose the cluster used for training.
        """

        self.use_half = use_half

        self.init = (mode == 1)
        self.sparse = sparse
        self.eye = (mode == 2)

        self.image_height = image_height

        self.augment = augment
        self.aug_rotation = aug_rotation
        self.aug_scale_min = aug_scale_min
        self.aug_scale_max = aug_scale_max
        self.aug_black_white = aug_black_white
        self.aug_color = aug_color

        self.num_clusters = num_clusters
        self.cluster_idx = cluster_idx

        if self.num_clusters is not None:
            if self.num_clusters < 1:
                raise ValueError("num_clusters must be at least 1")

            if self.cluster_idx is None:
                raise ValueError("cluster_idx needs to be specified when num_clusters is set")

            if self.cluster_idx < 0 or self.cluster_idx >= self.num_clusters:
                raise ValueError(f"cluster_idx needs to be between 0 and {self.num_clusters - 1}")

        if self.eye and self.augment and (self.aug_rotation > 0 or self.aug_scale_min != 1 or self.aug_scale_max != 1):
            # pre-generated eye coordinates cannot be augmented
            _logger.warning("WARNING: Check your augmentation settings. Camera coordinates will not be augmented.")

        # Setup data paths.
        root_dir = Path(root_dir)

        # Main folders.
        rgb_dir = root_dir / 'rgb'
        pose_dir = root_dir / 'poses'
        calibration_dir = root_dir / 'calibration'

        # Optional folders. Unused in ACE.
        if self.eye:
            coord_dir = root_dir / 'eye'
        elif self.sparse:
            coord_dir = root_dir / 'init'
        else:
            coord_dir = root_dir / 'depth'

        # Find all images. The assumption is that it only contains image files.
        self.rgb_files = sorted(rgb_dir.iterdir(), key=lambda x: int(x.stem))

        # Load camera calibrations. One focal length per image.
        self.calibration_files = sorted(calibration_dir.iterdir(), key=lambda x: int(x.stem))

        if self.init or self.eye:
            # Load GT scene coordinates.
            self.coord_files = sorted(coord_dir.iterdir(), key=lambda x: int(x.stem))
        else:
            self.coord_files = None

        if len(self.rgb_files) != len(self.calibration_files):
            raise RuntimeError('RGB file count does not match calibration file count!')

        if self.coord_files and len(self.rgb_files) != len(self.coord_files):
            raise RuntimeError('RGB file count does not match coordinate file count!')

        # Create grid of 2D pixel positions used when generating scene coordinates from depth.
        if self.init and not self.sparse:
            self.prediction_grid = self._create_prediction_grid()
        else:
            self.prediction_grid = None

        # Image transformations. Excluding scale since that can vary batch-by-batch.
        if self.augment:
            self.image_transform = transforms.Compose([
                # transforms.ToPILImage(),
                # transforms.Resize(int(self.image_height * scale_factor)),
                transforms.Grayscale(),
                transforms.ColorJitter(brightness=self.aug_black_white, contrast=self.aug_black_white),
                # saturation=self.aug_color, hue=self.aug_color),  # Disable colour augmentation.
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4],  # statistics calculated over 7scenes training set, should generalize fairly well
                    std=[0.25]
                ),
            ])
        else:
            self.image_transform = transforms.Compose([
                # transforms.ToPILImage(),
                # transforms.Resize(self.image_height),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4],  # statistics calculated over 7scenes training set, should generalize fairly well
                    std=[0.25]
                ),
            ])

        # We use this to iterate over all frames. If clustering is enabled this is used to filter them.
        self.valid_file_indices = np.arange(len(self.rgb_files))

        # If clustering is enabled.
        if self.num_clusters is not None:
            _logger.info(f"Clustering the {len(self.rgb_files)} into {num_clusters} clusters.")
            _, _, cluster_labels = self._cluster(num_clusters)

            self.valid_file_indices = np.flatnonzero(cluster_labels == cluster_idx)
            _logger.info(f"After clustering, chosen cluster: {cluster_idx}, Using {len(self.valid_file_indices)} images.")


    @staticmethod
    def _create_prediction_grid():
        # Assumes all input images have a resolution smaller than 5000x5000.
        prediction_grid = np.zeros((2,
                                    math.ceil(5000 / Regressor.OUTPUT_SUBSAMPLE),
                                    math.ceil(5000 / Regressor.OUTPUT_SUBSAMPLE)))

        for x in range(0, prediction_grid.shape[2]):
            for y in range(0, prediction_grid.shape[1]):
                prediction_grid[0, y, x] = x * Regressor.OUTPUT_SUBSAMPLE
                prediction_grid[1, y, x] = y * Regressor.OUTPUT_SUBSAMPLE

        return prediction_grid

    @staticmethod
    def _resize_image(image, image_height):
        # Resize a numpy image as PIL. Works slightly better than resizing the tensor using torch's internal function.
        image = TF.to_pil_image(image)
        image = TF.resize(image, image_height)
        return image

    @staticmethod
    def _rotate_image(image, angle, order, mode='constant'):
        # Image is a torch tensor (CxHxW), convert it to numpy as HxWxC.
        image = image.permute(1, 2, 0).numpy()
        # Apply rotation.
        image = rotate(image, angle, order=order, mode=mode)
        # Back to torch tensor.
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        return image

    def _load_image(self, idx):
        image = io.imread(self.rgb_files[idx])

        if len(image.shape) < 3:
            # Convert to RGB if needed.
            image = color.gray2rgb(image)

        return image

    def _get_single_item(self, idx, image_height):
        # Apply index indirection.
        idx = self.valid_file_indices[idx]

        # Load image.
        image = self._load_image(idx)

        # Load intrinsics.
        k = np.loadtxt(self.calibration_files[idx])
        if k.size == 1:
            focal_length = float(k)
            centre_point = None
        elif k.shape == (3, 3):
            k = k.tolist()
            focal_length = [k[0][0], k[1][1]]
            centre_point = [k[0][2], k[1][2]]
        else: 
            raise Exception("Calibration file must contain either a 3x3 camera \
                intrinsics matrix or a single float giving the focal length \
                of the camera.")

        # The image will be scaled to image_height, adjust focal length as well.
        f_scale_factor = image_height / image.shape[0]
        if centre_point:
            centre_point = [c * f_scale_factor for c in centre_point]
            focal_length = [f * f_scale_factor for f in focal_length]
        else:
            focal_length *= f_scale_factor

        # Rescale image.
        image = self._resize_image(image, image_height)
        # Create mask of the same size as the resized image (it's a PIL image at this point).
        image_mask = torch.ones((1, image.shape[1], image.shape[2]))
        # Apply remaining transforms.
        image = self.image_transform(image)


        # Load ground truth scene coordinates, if needed.
        if self.init:
            if self.sparse:
                coords = torch.load(self.coord_files[idx])
            else:
                depth = io.imread(self.coord_files[idx])
                depth = depth.astype(np.float64)
                depth /= 1000  # from millimeters to meters
        elif self.eye:
            coords = torch.load(self.coord_files[idx])
        else:
            coords = 0  # Default for ACE, we don't need them.


        # Convert to half if needed.
        if self.use_half and torch.cuda.is_available():
            image = image.half()

        # Binarize the mask.
        image_mask = image_mask > 0

        # Create the intrinsics matrix.
        intrinsics = torch.eye(3)
        
        # Hardcode the principal point to the centre of the image unless otherwise specified.
        if centre_point:
            intrinsics[0, 0] = focal_length[0]
            intrinsics[1, 1] = focal_length[1]
            intrinsics[0, 2] = centre_point[0]
            intrinsics[1, 2] = centre_point[1]
        else:
            intrinsics[0, 0] = focal_length
            intrinsics[1, 1] = focal_length
            intrinsics[0, 2] = image.shape[2] / 2
            intrinsics[1, 2] = image.shape[1] / 2

        # Also need the inverse.
        intrinsics_inv = intrinsics.inverse()

        # return image, image_mask, intrinsics, intrinsics_inv, coords, str(self.rgb_files[idx])
        out = {}
        out['image'] = image
        out['image_mask'] = image_mask
        out['intrinsics'] = intrinsics
        out['intrinsics_inv'] = intrinsics_inv
        out['scene_coords'] = coords
        out['rgb_files'] = str(self.rgb_files[idx])
        return out

    def __len__(self):
        return len(self.valid_file_indices)

    def __getitem__(self, idx):
        if self.augment:
            scale_factor = random.uniform(self.aug_scale_min, self.aug_scale_max)
            # scale_factor = 1 / scale_factor #inverse scale sampling, not used for ACE mapping
        else:
            scale_factor = 1

        # Target image height. We compute it here in case we are asked for a full batch of tensors because we need
        # to apply the same scale factor to all of them.
        image_height = int(self.image_height * scale_factor)

        if type(idx) == list:
            # Whole batch.
            tensors = [self._get_single_item(i, image_height) for i in idx]
            return default_collate(tensors)
        else:
            # Single element.
            return self._get_single_item(idx, image_height)
