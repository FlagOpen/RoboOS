#!/usr/bin/env bash

# Find the path to the root of the repo.
SCRIPT_PATH=$(dirname $(realpath -s "$0"))
REPO_PATH=$(realpath -s "${SCRIPT_PATH}/..")

scenes=("7scenes_chess" "7scenes_fire" "7scenes_heads" "7scenes_office" "7scenes_pumpkin" "7scenes_redkitchen" "7scenes_stairs")

training_exe="${REPO_PATH}/train_ace.py"
testing_exe="${REPO_PATH}/test_ace.py"

datasets_folder="${REPO_PATH}/datasets"
out_dir="${REPO_PATH}/output/viz_maps/7Scenes"
renderings_dir="${REPO_PATH}/output/renderings/7Scenes"

mkdir -p "$out_dir"
mkdir -p "renderings_dir"

for scene in ${scenes[*]}; do
  python $training_exe "$datasets_folder/$scene" "$out_dir/$scene.pt" --render_visualization True --render_target_path "$renderings_dir" --render_camera_z_offset 2
  python $testing_exe "$datasets_folder/$scene" "$out_dir/$scene.pt" --render_visualization True  --render_target_path "$renderings_dir" --render_pose_error_threshold 10 --render_frame_skip 3
  /usr/bin/ffmpeg -framerate 30 -pattern_type glob -i "$renderings_dir/$scene/*.png" -c:v libx264 -pix_fmt yuv420p "$renderings_dir/$scene.mp4"
done