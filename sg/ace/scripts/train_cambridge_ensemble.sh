#!/usr/bin/env bash

# Find the path to the root of the repo.
SCRIPT_PATH=$(dirname $(realpath -s "$0"))
REPO_PATH=$(realpath -s "${SCRIPT_PATH}/..")

scenes=("Cambridge_GreatCourt" "Cambridge_KingsCollege" "Cambridge_OldHospital" "Cambridge_ShopFacade" "Cambridge_StMarysChurch")

training_exe="${REPO_PATH}/train_ace.py"
testing_exe="${REPO_PATH}/test_ace.py"
merging_exe="${REPO_PATH}/merge_ensemble_results.py"
eval_exe="${REPO_PATH}/eval_poses.py"

datasets_folder="${REPO_PATH}/datasets"
out_dir="${REPO_PATH}/output/Cambridge_Ensemble"
mkdir -p "$out_dir"

num_clusters=${1:-1}

for scene in ${scenes[*]}; do
  current_out_dir="$out_dir/$scene"
  mkdir -p "$current_out_dir"

  for cluster_idx in $(seq 0 $((num_clusters-1))); do
    echo "Training network for scene: $scene and cluster: $cluster_idx/$num_clusters"

    net_file="${current_out_dir}/${cluster_idx}_${num_clusters}.pt"

    [[ -f "$net_file" ]] || python $training_exe "$datasets_folder/$scene" "$net_file" --num_clusters $num_clusters --cluster_idx $cluster_idx | tee "${current_out_dir}/train_${cluster_idx}_${num_clusters}.txt"
    python $testing_exe "$datasets_folder/$scene" "$net_file" --session "${cluster_idx}_${num_clusters}" 2>&1 | tee "${current_out_dir}/test_${cluster_idx}_${num_clusters}.txt"
  done

  python $merging_exe "$current_out_dir" "$current_out_dir/merged_poses_${num_clusters}.txt" --poses_suffix "_${num_clusters}.txt"
  python $eval_exe "$datasets_folder/$scene" "$current_out_dir/merged_poses_${num_clusters}.txt" 2>&1 | tee "$current_out_dir/eval_merged_${num_clusters}.txt"
done

for scene in ${scenes[*]}; do
  current_out_dir="$out_dir/$scene"
  echo "${scene}: $(cat "$current_out_dir/eval_merged_${num_clusters}.txt" | tail -1)"
done
