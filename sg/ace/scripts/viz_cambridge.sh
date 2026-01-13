#!/usr/bin/env bash

# Find the path to the root of the repo.
SCRIPT_PATH=$(dirname $(realpath -s "$0"))
REPO_PATH=$(realpath -s "${SCRIPT_PATH}/..")

scenes=( "Cambridge_OldHospital" "Cambridge_ShopFacade" "Cambridge_KingsCollege" "Cambridge_GreatCourt" "Cambridge_StMarysChurch")

training_exe="${REPO_PATH}/train_ace.py"
testing_exe="${REPO_PATH}/test_ace.py"

datasets_folder="${REPO_PATH}/datasets"
out_dir="${REPO_PATH}/output/viz_maps/Cambridge"
renderings_dir="${REPO_PATH}/output/renderings/Cambridge"

mkdir -p "$out_dir"

for scene in ${scenes[*]}; do
  python $training_exe "$datasets_folder/$scene" "$out_dir/$scene.pt" --render_visualization True --render_target_path "$renderings_dir" --render_map_error_threshold 50 --render_map_depth_filter 50
  python $testing_exe "$datasets_folder/$scene" "$out_dir/$scene.pt" --render_visualization True --render_target_path "$renderings_dir" --render_sparse_queries True --render_pose_error_threshold 100 --render_map_depth_filter 50 --render_camera_z_offset 8
  /usr/bin/ffmpeg -framerate 30 -pattern_type glob -i "$renderings_dir/$scene/*.png" -c:v libx264 -pix_fmt yuv420p "$renderings_dir/$scene.mp4"
done