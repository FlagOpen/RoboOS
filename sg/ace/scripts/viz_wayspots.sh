#!/usr/bin/env bash

# Find the path to the root of the repo.
SCRIPT_PATH=$(dirname $(realpath -s "$0"))
REPO_PATH=$(realpath -s "${SCRIPT_PATH}/..")

scenes=("wayspots_squarebench" "wayspots_bears" "wayspots_cubes" "wayspots_inscription" "wayspots_lawn" "wayspots_map"  "wayspots_statue" "wayspots_tendrils" "wayspots_therock" "wayspots_wintersign")

training_exe="${REPO_PATH}/train_ace.py"
testing_exe="${REPO_PATH}/test_ace.py"

datasets_folder="${REPO_PATH}/datasets"
out_dir="${REPO_PATH}/output/viz_maps/wayspots"
renderings_dir="${REPO_PATH}/output/renderings/wayspots"

mkdir -p "$out_dir"

for scene in ${scenes[*]}; do
  python $training_exe "$datasets_folder/$scene" "$out_dir/$scene.pt" --render_visualization True --render_target_path "$renderings_dir" --render_flipped_portrait True
  python $testing_exe "$datasets_folder/$scene" "$out_dir/$scene.pt" --render_visualization True --render_target_path "$renderings_dir" --render_flipped_portrait True
  /usr/bin/ffmpeg -framerate 30 -pattern_type glob -i "$renderings_dir/$scene/*.png" -c:v libx264 -pix_fmt yuv420p "$renderings_dir/$scene.mp4"
done

