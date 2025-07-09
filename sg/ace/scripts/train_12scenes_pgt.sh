#!/usr/bin/env bash

# Find the path to the root of the repo.
SCRIPT_PATH=$(dirname $(realpath -s "$0"))
REPO_PATH=$(realpath -s "${SCRIPT_PATH}/..")

scenes=("pgt_12scenes_apt1_kitchen" "pgt_12scenes_apt1_living" "pgt_12scenes_apt2_bed" "pgt_12scenes_apt2_kitchen" "pgt_12scenes_apt2_living" "pgt_12scenes_apt2_luke" "pgt_12scenes_office1_gates362" "pgt_12scenes_office1_gates381" "pgt_12scenes_office1_lounge" "pgt_12scenes_office1_manolis" "pgt_12scenes_office2_5a" "pgt_12scenes_office2_5b")

training_exe="${REPO_PATH}/train_ace.py"
testing_exe="${REPO_PATH}/test_ace.py"

datasets_folder="${REPO_PATH}/datasets"
out_dir="${REPO_PATH}/output/pgt_12Scenes"
mkdir -p "$out_dir"

for scene in ${scenes[*]}; do
  python $training_exe "$datasets_folder/$scene" "$out_dir/$scene.pt"
  python $testing_exe "$datasets_folder/$scene" "$out_dir/$scene.pt" 2>&1 | tee "$out_dir/log_${scene}.txt"
done

for scene in ${scenes[*]}; do
  echo "${scene}: $(cat "${out_dir}/log_${scene}.txt" | tail -5 | head -1)"
done
