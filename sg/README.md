# Scene Graph Setup and Execution Guide

## 1. Setup DROID-SLAM Environment

```bash
cd third_party/DROID-SLAM

# Create and activate conda environment
conda create -n droidenv python=3.9 -y
conda activate droidenv

# Install PyTorch 1.10 with CUDA 11.3
conda install pytorch=1.10 torchvision torchaudio cudatoolkit=11.3 -c pytorch -y

# Install additional libraries
conda install suitesparse -c conda-forge -y
pip install open3d==0.15.2 scipy opencv-python==4.7.0.72 matplotlib pyyaml==6.0.2 tensorboard
pip install evo --upgrade --no-binary evo
pip install gdown
pip install numpy==1.23.0 numpy-quaternion==2023.0.4

# Install torch_scatter
wget https://data.pyg.org/whl/torch-1.10.0%2Bcu113/torch_scatter-2.0.9-cp39-cp39-linux_x86_64.whl
pip install torch_scatter-2.0.9-cp39-cp39-linux_x86_64.whl
```

### Optional Fixes

If your system lacks `gcc-10`:

```bash
# Use system or conda compiler
sudo apt install gcc-10 g++-10
# OR
conda install gcc=10 gxx=10
```

If encountering `libtorch_cpu.so` executable stack error:

```bash
sudo apt install execstack
execstack -c $CONDA_PREFIX/lib/python3.9/site-packages/torch/lib/libtorch_cpu.so
```

Set CUDA environment variables:

```bash
export CUDA_HOME=/usr/local/cuda-11.3
export PATH=/usr/local/cuda-11.3/bin:$PATH
```

Build the project:

```bash
python setup.py install
```

---

## 2. Setup SG Environment

```bash
cd ../../sg

# Create and activate conda environment
conda create -n sg python=3.9 -y
conda activate sg

# Install PyTorch 2.3.1 with CUDA 11.8
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118

# Install requirements
pip install -r requirements.txt
```

### Install Third-Party Dependencies

```bash
cd ../third_party/segment-anything-2
pip install -e .
pip install -e ".[demo]"

cd ../GroundingDINO
pip install -e .

cd ../
pip install -r ./recognize-anything/requirements.txt
pip install -e ./recognize-anything/

sudo apt-get install libopencv-dev
conda install opencv
```

### Build DSAC\* and LightGlue

```bash
cd ../sg/ace/dsacstar
python setup.py install

cd ../../../third_party/LightGlue
python -m pip install -e .

conda install -c pytorch faiss-cpu=1.7.4 mkl=2021 blas=1.0=mkl
conda install -c conda-forge libopenblas libcblas
```

### Install PyTorch3D

```bash
cd ../../third_party/pytorch3d
python setup.py install

cd ../..
pip install -e .
```

---

## 3. Prepare Model Checkpoints

```bash
cd sg
mkdir checkpoints
cd checkpoints

# Clone required models
git lfs install
git clone https://huggingface.co/google-bert/bert-base-uncased
git clone https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K

# Manually download DROID-SLAM checkpoint from:
# https://drive.google.com/file/d/1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh/view

mkdir GroundingDINO segment-anything-2 recognize_anything

# Download pretrained models
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -O GroundingDINO/groundingdino_swint_ogc.pth
wget https://huggingface.co/spaces/xinyu1205/recognize-anything/resolve/main/ram_swin_large_14m.pth -O recognize_anything/ram_swin_large_14m.pth
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt -O segment-anything-2/sam2_hiera_large.pt

# Install Graphviz
sudo apt install graphviz
```

---

## 4. Realsense Camera Setup (L515)

```bash
sudo mkdir -p /etc/apt/keyrings
curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | sudo gpg --dearmor -o /etc/apt/keyrings/librealsense.gpg
echo "deb [signed-by=/etc/apt/keyrings/librealsense.gpg] https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/librealsense.list

sudo apt-get update

# Install version 2.53.1 for L515 compatibility
sudo apt-get install --allow-downgrades \
  librealsense2=2.53.1-0~realsense0.8251 \
  librealsense2-dev=2.53.1-0~realsense0.8251 \
  librealsense2-utils=2.53.1-0~realsense0.8251 \
  librealsense2-gl=2.53.1-0~realsense0.8251 \
  librealsense2-net=2.53.1-0~realsense0.8251 \
  librealsense2-udev-rules=2.53.1-0~realsense0.8251 \
  librealsense2-dkms=1.3.19-0ubuntu1

# Test camera
realsense-viewer
```

---

## 5. Running the Full Pipeline

### Step 1: Record Data with RGB-D Camera

```bash
conda activate sg

python demo.py \
    --tags "office_test" \
    --scanning_room \
    --preprocess \
    --task_scene_change_level "Minor Adjustment" \
    --rs_serial_number "f1421695"
```

### Step 2: Run DROID-SLAM Pose Estimation

```bash
conda deactivate
conda activate droidenv

python scripts/pose_estimation.py \
    --datadir "data_example/office_test" \
    --calib "data_example/office_test/calib.txt" \
    --t0 0 \
    --stride 1 \
    --weights "checkpoints/droid-slam/droid.pth" \
    --buffer 2048 \
    --depth_ratio 4000
```

### Step 3: Visualize Point Cloud

```bash
conda deactivate
conda activate sg

python scripts/show_pointcloud.py \
    --tags "office_test" \
    --pose_tags "poses_droidslam"
```

### Step 4: Run Demo

```bash
python demo.py \
    --tags "office_test" \
    --preprocess \
    --debug \
    --task_scene_change_level "Minor Adjustment" \
    --rs_serial_number "f1421695"
```
If crashing occurs, try:

```bash
pip install open3d==0.19.0
```
---

## Notes

* If any issue with `numpy.float` occurs, fix by:

  ```bash
  conda install numpy==1.23.0
  ```
The code is adopted from https://github.com/princeton-vl/DROID-SLAM, https://github.com/IDEA-Research/GroundingDINO, https://github.com/cvg/LightGlue, https://github.com/facebookresearch/pytorch3d, https://github.com/xinyu1205/recognize-anything, https://github.com/facebookresearch/sam2, and https://github.com/BJHYZJ/DovSG, etc.