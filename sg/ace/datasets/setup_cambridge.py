#!/usr/bin/env python3

import argparse
import math
import os

import cv2 as cv
import dataset_util as dutil
import numpy as np
import torch
from skimage import io

# setup individual scene IDs and their download location
scenes = [
    'https://www.repository.cam.ac.uk/bitstream/handle/1810/251342/KingsCollege.zip',
    'https://www.repository.cam.ac.uk/bitstream/handle/1810/251340/OldHospital.zip',
    'https://www.repository.cam.ac.uk/bitstream/handle/1810/251336/ShopFacade.zip',
    'https://www.repository.cam.ac.uk/bitstream/handle/1810/251294/StMarysChurch.zip',
    'https://www.repository.cam.ac.uk/bitstream/handle/1810/251291/GreatCourt.zip',
]

target_height = 480  # rescale images
nn_subsampling = 8  # sub sampling of our CNN architecture, for size of the initalization targets

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Download and setup the Cambridge dataset.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--init', type=str, choices=['none', 'sfm'], default='none',
                        help='none: no initialisation targets for scene coordinates; sfm: scene coordinate targets by rendering the SfM point cloud')

    opt = parser.parse_args()

    print("\n###############################################################################")
    print("# Please make sure to check this dataset's license before using it!           #")
    print("# https://www.repository.cam.ac.uk/items/53788265-cb98-42ee-b85b-7a0cbc8eddb3 #")
    print("###############################################################################\n\n")

    license_response = input('Please confirm with "yes" or abort. ')
    if not (license_response == "yes" or license_response == "y"):
        print(f"Your response: {license_response}. Aborting.")
        exit()

    for scene in scenes:

        scene_file = scene.split('/')[-1]
        scene_name = scene_file[:-4]

        print("===== Processing " + scene_name + " ===================")

        print("Downloading and unzipping data...")
        os.system('wget ' + scene)
        os.system('unzip ' + scene_file)
        os.system('rm ' + scene_file)
        os.system('mv ' + scene_name + ' Cambridge_' + scene_name)
        os.chdir('Cambridge_' + scene_name)

        modes = ['train', 'test']
        input_file = 'reconstruction.nvm'

        print("Loading SfM reconstruction...")

        f = open(input_file)
        reconstruction = f.readlines()
        f.close()

        num_cams = int(reconstruction[2])
        num_pts = int(reconstruction[num_cams + 4])

        if opt.init == 'sfm':

            # read points
            pts_dict = {}
            for cam_idx in range(0, num_cams):
                pts_dict[cam_idx] = []

            pt = pts_start = num_cams + 5
            pts_end = pts_start + num_pts

            while pt < pts_end:

                pt_list = reconstruction[pt].split()
                pt_3D = [float(x) for x in pt_list[0:3]]
                pt_3D.append(1.0)

                for pt_view in range(0, int(pt_list[6])):
                    cam_view = int(pt_list[7 + pt_view * 4])
                    pts_dict[cam_view].append(pt_3D)

                pt += 1

        print("Reconstruction contains %d cameras and %d 3D points." % (num_cams, num_pts))

        for mode in modes:

            print("Converting " + mode + " data...")

            img_output_folder = mode + '/rgb/'
            cal_output_folder = mode + '/calibration/'
            pose_output_folder = mode + '/poses/'

            dutil.mkdir(img_output_folder)
            dutil.mkdir(cal_output_folder)
            dutil.mkdir(pose_output_folder)

            if opt.init != 'none':
                target_output_folder = mode + '/init/'
                dutil.mkdir(target_output_folder)

            # get list of images for current mode (train vs. test)
            image_list = 'dataset_' + mode + '.txt'

            f = open(image_list)
            camera_list = f.readlines()
            f.close()
            camera_list = camera_list[3:]

            image_list = [camera.split()[0] for camera in camera_list]

            for cam_idx in range(num_cams):

                print("Processing camera %d of %d." % (cam_idx, num_cams))
                image_file = reconstruction[3 + cam_idx].split()[0]
                image_file = image_file[:-3] + 'png'

                if image_file not in image_list:
                    print("Skipping image " + image_file + ". Not part of set: " + mode + ".")
                    continue

                image_idx = image_list.index(image_file)

                # read camera
                camera = camera_list[image_idx].split()
                cam_rot = [float(r) for r in camera[4:]]

                # quaternion to axis-angle
                angle = 2 * math.acos(cam_rot[0])
                x = cam_rot[1] / math.sqrt(1 - cam_rot[0] ** 2)
                y = cam_rot[2] / math.sqrt(1 - cam_rot[0] ** 2)
                z = cam_rot[3] / math.sqrt(1 - cam_rot[0] ** 2)

                cam_rot = [x * angle, y * angle, z * angle]

                cam_rot = np.asarray(cam_rot)
                cam_rot, _ = cv.Rodrigues(cam_rot)

                cam_trans = [float(r) for r in camera[1:4]]
                cam_trans = np.asarray([cam_trans])
                cam_trans = np.transpose(cam_trans)
                cam_trans = - np.matmul(cam_rot, cam_trans)

                if np.absolute(cam_trans).max() > 10000:
                    print("Skipping image " + image_file + ". Extremely large translation. Outlier?")
                    print(cam_trans)
                    continue

                cam_pose = np.concatenate((cam_rot, cam_trans), axis=1)
                cam_pose = np.concatenate((cam_pose, [[0, 0, 0, 1]]), axis=0)
                cam_pose = torch.tensor(cam_pose).float()

                focal_length = float(reconstruction[3 + cam_idx].split()[1])

                # load image
                image = io.imread(image_file)
                image_file = image_file.replace('/', '_')

                img_aspect = image.shape[0] / image.shape[1]

                if img_aspect > 1:
                    # portrait
                    img_w = target_height
                    img_h = int(math.ceil(target_height * img_aspect))
                else:
                    # landscape
                    img_w = int(math.ceil(target_height / img_aspect))
                    img_h = target_height

                out_w = int(math.ceil(img_w / nn_subsampling))
                out_h = int(math.ceil(img_h / nn_subsampling))

                out_scale = out_w / image.shape[1]
                img_scale = img_w / image.shape[1]

                image = cv.resize(image, (img_w, img_h))
                io.imsave(img_output_folder + image_file, image)

                with open(cal_output_folder + image_file[:-3] + 'txt', 'w') as f:
                    f.write(str(focal_length * img_scale))

                inv_cam_pose = cam_pose.inverse()

                with open(pose_output_folder + image_file[:-3] + 'txt', 'w') as f:
                    f.write(str(float(inv_cam_pose[0, 0])) + ' ' + str(float(inv_cam_pose[0, 1])) + ' ' + str(
                        float(inv_cam_pose[0, 2])) + ' ' + str(float(inv_cam_pose[0, 3])) + '\n')
                    f.write(str(float(inv_cam_pose[1, 0])) + ' ' + str(float(inv_cam_pose[1, 1])) + ' ' + str(
                        float(inv_cam_pose[1, 2])) + ' ' + str(float(inv_cam_pose[1, 3])) + '\n')
                    f.write(str(float(inv_cam_pose[2, 0])) + ' ' + str(float(inv_cam_pose[2, 1])) + ' ' + str(
                        float(inv_cam_pose[2, 2])) + ' ' + str(float(inv_cam_pose[2, 3])) + '\n')
                    f.write(str(float(inv_cam_pose[3, 0])) + ' ' + str(float(inv_cam_pose[3, 1])) + ' ' + str(
                        float(inv_cam_pose[3, 2])) + ' ' + str(float(inv_cam_pose[3, 3])) + '\n')

                if opt.init == 'sfm':

                    # load 3D points from reconstruction
                    pts_3D = torch.tensor(pts_dict[cam_idx])

                    out_tensor = torch.zeros((3, out_h, out_w))
                    out_zbuffer = torch.zeros((out_h, out_w))

                    fine = 0
                    conflict = 0

                    for pt_idx in range(0, pts_3D.size(0)):

                        scene_pt = pts_3D[pt_idx]
                        scene_pt = scene_pt.unsqueeze(0)
                        scene_pt = scene_pt.transpose(0, 1)

                        # scene to camera coordinates
                        cam_pt = torch.mm(cam_pose, scene_pt)
                        # projection to image
                        img_pt = cam_pt[0:2, 0] * focal_length / cam_pt[2, 0] * out_scale

                        y = img_pt[1] + out_h / 2
                        x = img_pt[0] + out_w / 2

                        x = int(torch.clamp(x, min=0, max=out_tensor.size(2) - 1))
                        y = int(torch.clamp(y, min=0, max=out_tensor.size(1) - 1))

                        if cam_pt[2, 0] > 1000:  # filter some outlier points (large depth)
                            continue

                        if out_zbuffer[y, x] == 0 or out_zbuffer[y, x] > cam_pt[2, 0]:
                            out_zbuffer[y, x] = cam_pt[2, 0]
                            out_tensor[:, y, x] = pts_3D[pt_idx, 0:3]

                    torch.save(out_tensor, target_output_folder + image_file[:-4] + '.dat')

        os.chdir('..')
