# Accelerated Coordinate Encoding: Learning to Relocalize in Minutes using RGB and Poses

----------------------------------------------------------------------------------------

This repository contains the code associated to the ACE paper:
> **Accelerated Coordinate Encoding: Learning to Relocalize in Minutes using RGB and Poses**
> 
> [Eric Brachmann](https://ebrach.github.io/), [Tommaso Cavallari](https://scholar.google.it/citations?user=r7osSm0AAAAJ&hl=en), and [Victor Adrian Prisacariu](https://www.robots.ox.ac.uk/~victor/)
> 
> [CVPR 2023, Highlight](https://openaccess.thecvf.com/content/CVPR2023/papers/Brachmann_Accelerated_Coordinate_Encoding_Learning_to_Relocalize_in_Minutes_Using_RGB_CVPR_2023_paper.pdf)

For further information please visit:

- [Project page (with videos, method explanations, dataset details)](https://nianticlabs.github.io/ace)
- [Arxiv](https://arxiv.org/abs/2305.14059)

Table of contents:

- [Installation](#installation)
- [Dataset Setup](#datasets)
- [Usage](#usage)
    - [ACE Training](#ace-training)
    - [ACE Evaluation](#ace-evaluation)
    - [Training Scripts](#complete-training-and-evaluation-scripts)
    - [Pretrained ACE Networks](#pretrained-ace-networks)
    - [Note on the Encoder Training](#encoder-training)
- [References](#publications)


## Installation

This code uses PyTorch to train and evaluate the scene-specific coordinate prediction head networks. It has been tested
on Ubuntu 20.04 with a T4 Nvidia GPU, although it should reasonably run with other Linux distributions and GPUs as well.

We provide a pre-configured [`conda`](https://docs.conda.io/en/latest/) environment containing all required dependencies
necessary to run our code.
You can re-create and activate the environment with:

```shell
conda env create -f environment.yml
conda activate ace
```

**All the following commands in this file need to run in the `ace` environment.**

The ACE network predicts dense 3D scene coordinates associated to the pixels of the input images.
In order to estimate the 6DoF camera poses, it relies on the RANSAC implementation of the DSAC* paper (Brachmann and
Rother, TPAMI 2021), which is written in C++.
As such, you need to build and install the C++/Python bindings of those functions.
You can do this with:

```shell
cd dsacstar
python setup.py install
```

> (Optional) If you want to create videos of the training/evaluation process:
> ```shell
> sudo apt install ffmpeg
> ```

Having done the steps above, you are ready to experiment with ACE!

## Datasets

The ACE method has been evaluated using multiple published datasets:

- [Microsoft 7-Scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/)
- [Stanford 12-Scenes](https://graphics.stanford.edu/projects/reloc/)
- [Cambridge Landmarks](https://www.repository.cam.ac.uk/handle/1810/251342/)
- [Niantic Wayspots](https://nianticlabs.github.io/ace#dataset)

We provide scripts in the `datasets` folder to automatically download and extract the data in a format that can be
readily used by the ACE scripts.
The format is the same used by the DSAC* codebase, see [here](https://github.com/vislearn/dsacstar#data-structure) for
details.

> **Important: make sure you have checked the license terms of each dataset before using it.**

### {7, 12}-Scenes:

You can use the `datasets/setup_{7,12}scenes.py` scripts to download the data.
As mentioned in the paper, we experimented with two variants of each of these datasets: one using the original
D-SLAM ground truth camera poses, and one using _Pseudo Ground Truth (PGT)_ camera poses obtained after running SfM on
the scenes
(see
the [ICCV 2021 paper](https://openaccess.thecvf.com/content/ICCV2021/html/Brachmann_On_the_Limits_of_Pseudo_Ground_Truth_in_Visual_Camera_ICCV_2021_paper.html)
,
and [associated code](https://github.com/tsattler/visloc_pseudo_gt_limitations/) for details).

To download and prepare the datasets using the D-SLAM poses:

```shell
cd datasets
# Downloads the data to datasets/7scenes_{chess, fire, ...}
./setup_7scenes.py
# Downloads the data to datasets/12scenes_{apt1_kitchen, ...}
./setup_12scenes.py
``` 

To download and prepare the datasets using the PGT poses:

```shell
cd datasets
# Downloads the data to datasets/pgt_7scenes_{chess, fire, ...}
./setup_7scenes.py --poses pgt
# Downloads the data to datasets/pgt_12scenes_{apt1_kitchen, ...}
./setup_12scenes.py --poses pgt
``` 

### Cambridge Landmarks / Niantic Wayspots:

We used a single variant of these datasets. Simply run:

```shell
cd datasets
# Downloads the data to datasets/Cambridge_{GreatCourt, KingsCollege, ...}
./setup_cambridge.py
# Downloads the data to datasets/wayspots_{bears, cubes, ...}
./setup_wayspots.py
```

## Usage

We provide scripts to train and evaluate ACE scene coordinate regression networks.
In the following sections we'll detail some of the main command line options that can be used to customize the
behavior of both the training and the pose estimation script.

### ACE Training

The ACE scene-specific coordinate regression head for a scene can be trained using the `train_ace.py` script.
Basic usage:

```shell
./train_ace.py <scene path> <output map name>
# Example:
./train_ace.py datasets/7scenes_chess output/7scenes_chess.pt
```

The output map file contains just the weights of the scene-specific head network -- encoded as half-precision floating
point -- for a size of ~4MB when using default options, as mentioned in the paper. The testing script will use these
weights, together with the scene-agnostic pretrained encoder (`ace_encoder_pretrained.pt`) we provide, to estimate 6DoF
poses for the query images.

**Additional parameters** that can be passed to the training script to alter its behavior:

- `--training_buffer_size`: Changes the size of the training buffer containing decorrelated image features (see paper),
  that is created at the beginning of the training process. The default size is 8M.
- `--samples_per_image`: How many features to sample from each image during the buffer generation phase. This affects
  the amount of time necessary to fill the training buffer, but also affects the amount of decorrelation in the features
  present in the buffer. The default is 1024 samples per image.
- `--epochs`: How many full passes over the training buffer are performed during the training. This directly affects the
  training time. Default is 16.
- `--num_head_blocks`: The depth of the head network. Specifically, the number of extra 3-layer residual blocks to add
  to the default head depth. Default value is 1, which results in a head network composed of 9 layers, for a total of
  4MB weights.

**Clustering parameters:** these are used for the ensemble experiments (ACE Poker variant) we ran on the Cambridge
Landmarks
dataset. They are used to split the input scene into multiple independent clusters, and training the head network on one
of them (see Section 4.2 of the main paper and Section 1.3 of the supplementary material for details).

- `--num_clusters`: How many clusters to split the training scene in. Default `None` (disabled).
- `--cluster_idx`: Selects a specific cluster for training.

**Visualization parameters:** these are used to generate the videos available in the project page (they actually
generate
individual frames that can be collated into a video later). _Note: enabling them will significantly slow down the
training._

- `--render_visualization`: Set to `True` to enable generating frames showing the training process. Default `False`.
- `--render_target_path`: Base folder where the frames will be saved. The script automatically appends the current map
  name to the folder. Default is `renderings`.

There are other options available, they can be discovered by running the script with the `--help` flag.

### ACE Evaluation

The pose estimation for a testing scene can be performed using the `test_ace.py` script.
Basic usage:

```shell
./test_ace.py <scene path> <output map name>
# Example:
./test_ace.py datasets/7scenes_chess output/7scenes_chess.pt
```

The script loads (a) the scene-specific ACE head network and (b) the pre-trained scene-agnostic encoder and, for each
testing frame:

- Computes its per-pixel 3D scene coordinates, resulting in a set of 2D-3D correspondences.
- The correspondences are then passed to a RANSAC algorithm that is able to estimate a 6DoF camera pose.
- The camera poses are compared with the ground truth, and various cumulative metrics are then computed and printed
  at the end of the script.

The metrics include: %-age of frames within certain translation/angle thresholds of the ground truth,
median translation, median rotation error.

The script also creates a file containing per-frame results so that they can be parsed by other tools or analyzed
separately.
The output file is located alongside the head network and is named: `poses_<map name>_<session>.txt`.

Each line in the output file contains the results for an individual query frame, in this format:

```
file_name rot_quaternion_w rot_quaternion_x rot_quaternion_y rot_quaternion_z translation_x translation_y translation_z rot_err_deg tr_err_m inlier_count
```

There are some parameters that can be passed to the script to customize the RANSAC behavior:

- `--session`: Custom suffix to append to the name of the file containing the estimated camera poses (see paragraph
  above).
- `--hypotheses`: How many pose hypotheses to generate and evaluate (i.e. the number of RANSAC iterations). Default is
    64.
- `--threshold`: Inlier threshold (in pixels) to consider a 2D-3D correspondence as valid.
- `--render_visualization`: Set to `True` to enable generating frames showing the evaluation process. Will slow down the
  testing significantly if enabled. Default `False`.
- `--render_target_path`: Base folder where the frames will be saved. The script automatically appends the current map
  name to the folder. Default is `renderings`.

There are other options available, they can be discovered by running the script with the `--help` flag.

#### Ensemble Evaluation

To deploy the ensemble variants (such as the 4-cluster ACE Poker variant), we simply need to run the training script
multiple times (once per cluster), thus training multiple head networks.

At localization time we run the testing script once for each trained head and save the per-frame `poses_...` files,
passing the `--session` parameter to the test script to tag each file according to the network used to generate it.

We provide two more scripts:

1. `merge_ensemble_results.py`: Merge multiple `poses_` files -- choosing for each frame the pose that resulted in the
   best inlier count.
2. `eval_poses.py`: Compute the overall accuracy metrics (%-age of poses within threshold and median errors) that we showed
   in the paper.

ACE Poker example for a scene in the Cambridge dataset:

```shell
mkdir -p output/Cambridge_GreatCourt

# Head training:
./train_ace.py datasets/Cambridge_GreatCourt output/Cambridge_GreatCourt/0_4.pt --num_clusters 4 --cluster_idx 0
./train_ace.py datasets/Cambridge_GreatCourt output/Cambridge_GreatCourt/1_4.pt --num_clusters 4 --cluster_idx 1
./train_ace.py datasets/Cambridge_GreatCourt output/Cambridge_GreatCourt/2_4.pt --num_clusters 4 --cluster_idx 2
./train_ace.py datasets/Cambridge_GreatCourt output/Cambridge_GreatCourt/3_4.pt --num_clusters 4 --cluster_idx 3

# Per-cluster evaluation:
./test_ace.py datasets/Cambridge_GreatCourt output/Cambridge_GreatCourt/0_4.pt --session 0_4
./test_ace.py datasets/Cambridge_GreatCourt output/Cambridge_GreatCourt/1_4.pt --session 1_4
./test_ace.py datasets/Cambridge_GreatCourt output/Cambridge_GreatCourt/2_4.pt --session 2_4
./test_ace.py datasets/Cambridge_GreatCourt output/Cambridge_GreatCourt/3_4.pt --session 3_4

# Merging results and computing metrics.

# The merging script takes a --poses_suffix argument that's used to select only the 
# poses generated for the requested number of clusters. 
./merge_ensemble_results.py output/Cambridge_GreatCourt output/Cambridge_GreatCourt/merged_poses_4.txt --poses_suffix "_4.txt"

# The output poses output by the previous script are then evaluated against the scene ground truth data.
./eval_poses.py datasets/Cambridge_GreatCourt output/Cambridge_GreatCourt/merged_poses_4.txt
```

### Complete training and evaluation scripts

We provide several scripts to run training and evaluation on the various datasets we tested our method with.
These allow replicating the results we showcased in the paper.
They are located under the `scripts` folder: `scripts/train_*.sh`.

In the same folder we also provide scripts to generate videos of the training/testing protocol, as can be seen in the
project page. They are under `scripts/viz_*.sh`.

### Pretrained ACE Networks

We also make available the set of pretrained ACE Heads we used for the experiments in the paper.
Each head was trained for 5 minutes on one of the scenes in the various datasets, and was used to compute the accuracy
metrics we showed in the main text.

Each network can be passed directly to the `test_ace.py` script, together with the path to its dataset scene, to run
camera relocalization on the images of the testing split and compute the accuracy metrics, like this:

```shell
./test_ace.py datasets/7scenes_chess <Downloads>/7Scenes/7scenes_chess.pt
```

**The data is available
at [this location](https://storage.googleapis.com/niantic-lon-static/research/ace/ace_models.tar.gz).**

### Encoder Training

As mentioned above, in this repository we provide a set of weights for the pretrained feature extraction
backbone (`ace_encoder_pretrained.pt`) that was used in our experiments.
You are welcome to use them to experiment with ACE in novel scenes.
Both indoor and outdoor environments perform reasonably well with this set of weights, as shown in the paper.

**Unfortunately, we cannot provide the code to train the encoder as part of this release.**

The feature extractor has been trained on 100 scenes from [ScanNet](http://www.scan-net.org/), in parallel, for ~1 week,
as described in Section 3.3 of the paper and Section 1.1 of the supplementary material.
It is possible to reimplement the encoder training protocol following those instructions.

## Publications

If you use ACE or parts of its code in your own work, please cite:

```
@inproceedings{brachmann2023ace,
    title={Accelerated Coordinate Encoding: Learning to Relocalize in Minutes using RGB and Poses},
    author={Brachmann, Eric and Cavallari, Tommaso and Prisacariu, Victor Adrian},
    booktitle={CVPR},
    year={2023},
}
```

This code builds on previous camera relocalization pipelines, namely DSAC, DSAC++, and DSAC*. Please consider citing:

```
@inproceedings{brachmann2017dsac,
  title={{DSAC}-{Differentiable RANSAC} for Camera Localization},
  author={Brachmann, Eric and Krull, Alexander and Nowozin, Sebastian and Shotton, Jamie and Michel, Frank and Gumhold, Stefan and Rother, Carsten},
  booktitle={CVPR},
  year={2017}
}

@inproceedings{brachmann2018lessmore,
  title={Learning less is more - {6D} camera localization via {3D} surface regression},
  author={Brachmann, Eric and Rother, Carsten},
  booktitle={CVPR},
  year={2018}
}

@article{brachmann2021dsacstar,
  title={Visual Camera Re-Localization from {RGB} and {RGB-D} Images Using {DSAC}},
  author={Brachmann, Eric and Rother, Carsten},
  journal={TPAMI},
  year={2021}
}
```

## License

Copyright © Niantic, Inc. 2023. Patent Pending.
All rights reserved.
Please see the [license file](LICENSE) for terms.