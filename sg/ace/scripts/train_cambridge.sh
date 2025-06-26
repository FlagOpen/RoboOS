#!/usr/bin/env bash

# Find the path to the root of the repo.
SCRIPT_PATH=$(dirname $(realpath -s "$0"))
REPO_PATH=$(realpath -s "${SCRIPT_PATH}/..")

scenes=("Cambridge_GreatCourt" "Cambridge_KingsCollege" "Cambridge_OldHospital" "Cambridge_ShopFacade" "Cambridge_StMarysChurch")

training_exe="${REPO_PATH}/train_ace.py"
testing_exe="${REPO_PATH}/test_ace.py"

datasets_folder="${REPO_PATH}/datasets"
out_dir="${REPO_PATH}/output/Cambridge"

mkdir -p "$out_dir"

for scene in ${scenes[*]}; do
  python $training_exe "$datasets_folder/$scene" "$out_dir/$scene.pt"
  python $testing_exe "$datasets_folder/$scene" "$out_dir/$scene.pt" 2>&1 | tee "$out_dir/log_${scene}.txt"
done

for scene in ${scenes[*]}; do
  echo "${scene}: $(cat "${out_dir}/log_${scene}.txt" | tail -2 | head -1)"
done
