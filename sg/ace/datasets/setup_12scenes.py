#!/usr/bin/env python3

import argparse
import os
import zipfile

import dataset_util as dutil
import numpy as np
import torch
from joblib import Parallel, delayed
from skimage import io

# name of the folder where we download the original 12scenes dataset to
# we restructure the dataset by creating symbolic links to that folder
src_folder = '12scenes_source'

# sub sampling factor of eye coordinate tensor
nn_subsampling = 8

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Download and setup the 12Scenes dataset.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--depth', type=str, choices=['none', 'rendered', 'original'], default='none',
                        help='none: ignore depth maps; rendered: download depth rendered using 3D scene model (18GB), original: original depth maps')

    parser.add_argument('--eye', type=str, choices=['none', 'original'], default='none',
                        help='none: ignore eye coordinates; original: precompute eye coordinates from original depth maps')

    parser.add_argument('--poses', type=str, choices=['original', 'pgt'], default='original',
                        help='original: original pose files; '
                             'pgt: get SfM poses from external repository (Brachmann et al., ICCV21)')

    opt = parser.parse_args()

    if opt.depth == 'rendered' and opt.poses == 'pgt':
        print("Sorry. Rendered depth files are not compatible with SfM pose files, "
              "since both have missing frames and we would have to figure out the intersection. "
              "It can be done, but is not supported atm.")
        exit()

    print("\n#####################################################################")
    print("# Please make sure to check this dataset's license before using it! #")
    print("# http://graphics.stanford.edu/projects/reloc/                      #")
    print("#####################################################################\n\n")

    license_response = input('Please confirm with "yes" or abort. ')
    if not (license_response == "yes" or license_response == "y"):
        print(f"Your response: {license_response}. Aborting.")
        exit()

    if opt.poses == 'pgt':

        print("\n###################################################################")
        print("# You requested external pose files. Please check the license at: #")
        print("# https://github.com/tsattler/visloc_pseudo_gt_limitations        #")
        print("###################################################################\n\n")

        license_response = input('Please confirm with "yes" or abort. ')
        if not (license_response == "yes" or license_response == "y"):
            print(f"Your response: {license_response}. Aborting.")
            exit()

        print("Getting external pose files...")
        external_pgt_folder = dutil.clone_external_pose_files()

    # download the original 12 scenes dataset for calibration, poses and images
    dutil.mkdir(src_folder)
    os.chdir(src_folder)

    for ds in ['apt1', 'apt2', 'office1', 'office2']:

        if not os.path.exists(ds):

            print("=== Downloading 12scenes Data:", ds, "===============================")

            os.system('wget http://graphics.stanford.edu/projects/reloc/data/' + ds + '.zip')

            # unpack and delete zip file
            f = zipfile.PyZipFile(ds + '.zip')
            f.extractall()

            os.system('rm ' + ds + '.zip')

        else:
            print(f"Found data of scene {ds} already. Assuming its complete and skipping download.")


    def process_dataset(ds):

        scenes = os.listdir(ds)

        for scene in scenes:

            data_folder = ds + '/' + scene + '/data/'

            if not os.path.isdir(data_folder):
                # skip README files
                continue

            print("Linking files for 12scenes_" + ds + "_" + scene + "...")

            if opt.poses == 'pgt':
                target_folder = '../pgt_12scenes_' + ds + '_' + scene + '/'
            else:
                target_folder = '../12scenes_' + ds + '_' + scene + '/'

            # create subfolders for training and test
            dutil.mkdir(target_folder + 'test/rgb/')
            dutil.mkdir(target_folder + 'test/poses/')
            dutil.mkdir(target_folder + 'test/calibration/')
            if opt.depth == 'original':
                dutil.mkdir(target_folder + 'test/depth/')
            if opt.eye == 'original':
                dutil.mkdir(target_folder + 'test/eye/')

            dutil.mkdir(target_folder + 'train/rgb/')
            dutil.mkdir(target_folder + 'train/poses/')
            dutil.mkdir(target_folder + 'train/calibration/')
            if opt.depth == 'original':
                dutil.mkdir(target_folder + 'train/depth/')
            if opt.eye == 'original':
                dutil.mkdir(target_folder + 'train/eye/')

            # read the train / test split - the first sequence is used for testing, everything else for training
            with open(ds + '/' + scene + '/split.txt', 'r') as f:
                split = f.readlines()
            split = int(split[0].split()[1][8:-1])

            # read the calibration parameters
            with open(ds + '/' + scene + '/info.txt', 'r') as f:
                calibration_info = f.readlines()

            im_h = int(calibration_info[3].split()[2])

            focallength = calibration_info[7].split()
            focallength = (float(focallength[2]) + float(focallength[7])) / 2

            files = os.listdir(data_folder)

            images = [f for f in files if f.endswith('color.jpg')]
            images.sort()

            poses = [f for f in files if f.endswith('pose.txt')]
            poses.sort()

            if opt.depth == 'original' or opt.eye == 'original':
                depth_maps = [f for f in files if f.endswith('depth.png')]
                depth_maps.sort()

            # read external poses and calibration if requested
            if opt.poses == 'pgt':
                pgt_file = os.path.join('..', external_pgt_folder, '12scenes', f'{ds}_{scene}_test.txt')
                pgt_test_poses = dutil.read_pose_data(pgt_file)
                pgt_file = os.path.join('..', external_pgt_folder, '12scenes', f'{ds}_{scene}_train.txt')
                pgt_train_poses = dutil.read_pose_data(pgt_file)
            else:
                pgt_test_poses = None
                pgt_train_poses = None

            def link_frame(i, variant, pgt_poses):
                """ Links calibration, pose and image of frame i in either test or training. """

                # some image have invalid pose files, skip those
                if opt.poses == 'pgt':
                    valid = os.path.join('data', dutil.get_base_file_name(poses[i])) in pgt_poses
                else:

                    valid = True
                    with open(ds + '/' + scene + '/data/' + poses[i], 'r') as f:
                        pose = f.readlines()
                        for line in pose:
                            if 'INF' in line:
                                valid = False

                if not valid:
                    print("Skipping frame", i, "(" + variant + ") - Corrupt pose.")
                else:
                    # link image
                    os.system('ln -s ../../../' + src_folder + '/' + data_folder + '/' + images[
                        i] + ' ' + target_folder + variant + '/rgb/')

                    if opt.poses == 'pgt':
                        cam_pose, _ = pgt_poses[os.path.join('data', dutil.get_base_file_name(poses[i]))]
                        dutil.write_cam_pose(target_folder + variant + '/poses/' + poses[i], cam_pose)
                    else:
                        # link pose
                        os.system('ln -s ../../../' + src_folder + '/' + data_folder + '/' + poses[
                            i] + ' ' + target_folder + variant + '/poses/')

                    # create a calibration file
                    with open(target_folder + variant + '/calibration/frame-%s.calibration.txt' % str(i).zfill(6),
                              'w') as f:
                        f.write(str(focallength))

                    if opt.depth == 'original':
                        # link original depth files
                        os.system('ln -s ../../../' + src_folder + '/' + data_folder + '/' + depth_maps[
                            i] + ' ' + target_folder + variant + '/depth/')

                    if opt.eye == 'original':

                        depth = io.imread(data_folder + '/' + depth_maps[i])
                        depth = depth.astype(np.float64)
                        depth /= 1000  # from millimeters to meters

                        d_h = depth.shape[0]
                        d_w = depth.shape[1]

                        # get RGB focal length and adjust to depth resolution
                        if opt.poses == 'pgt':
                            _, rgb_f = pgt_poses[os.path.join('data', dutil.get_base_file_name(poses[i]))]
                        else:
                            rgb_f = focallength

                        d_f = rgb_f * d_h / im_h

                        # generate sub-sampled eye coordinate tensor from calibrated depth
                        out_h = int(d_h / nn_subsampling)
                        out_w = int(d_w / nn_subsampling)
                        nn_offset = int(nn_subsampling / 2)

                        eye_tensor = np.zeros((3, out_h, out_w))

                        # generate pixel coordinates
                        eye_tensor[0] = np.dstack([np.arange(0, out_w)] * out_h)[0].T * nn_subsampling + nn_offset
                        eye_tensor[1] = np.dstack([np.arange(0, out_h)] * out_w)[0] * nn_subsampling + nn_offset

                        # substract principal point
                        eye_tensor[0] -= d_w / 2
                        eye_tensor[1] -= d_h / 2

                        # subsample depth
                        depth = depth[nn_offset::nn_subsampling, nn_offset::nn_subsampling]

                        # project
                        eye_tensor[0:2] /= d_f
                        eye_tensor[2, 0:depth.shape[0], 0:depth.shape[1]] = depth
                        eye_tensor[0] *= eye_tensor[2]
                        eye_tensor[1] *= eye_tensor[2]

                        eye_tensor = torch.from_numpy(eye_tensor).float()

                        torch.save(eye_tensor, target_folder + variant + '/eye/' + depth_maps[i][:-10] + '.eye.dat')

            # frames up to split are test images
            for i in range(split):
                link_frame(i, 'test', pgt_test_poses)

            # all remaining frames are training images
            for i in range(split, len(images)):
                link_frame(i, 'train', pgt_train_poses)


    Parallel(n_jobs=4, verbose=0)(
        map(delayed(process_dataset), ['apt1', 'apt2', 'office1', 'office2']))

    if opt.depth == 'rendered':
        os.chdir('..')
        dutil.dlheidata("10.11588/data/N07HKC/OMLKR1", "12scenes_depth.tar.gz")
