#!/bin/bash
gpu="$1"

# Define the base command without the parameters we plan to vary
base_command="python src/iterative_cluster.py \
  --data_path processed_data/vicunadata_small \
  --subsample 0 \
  --proposer_model gpt-4 \
  --assigner_name google/flan-t5-xl \
  --proposer_num_descriptions_to_propose 100 \
  --assigner_for_final_assignment_template templates/t5_multi_assigner_one_output.txt \
  --cluster_num_clusters 10 \
  --iterative_max_rounds 1"

# Define ranges for the parameters
min_cluster_fractions=(0 0.01 0.02)
max_cluster_fractions=(0.15 0.2 0.3)
overlap_penalties=(0.1 0.2 0.3)
not_cover_penalties=(0.5 1.0)

# Loop through the ranges and run experiments
for min_cf in "${min_cluster_fractions[@]}"; do
  for max_cf in "${max_cluster_fractions[@]}"; do
    for overlap_penalty in "${overlap_penalties[@]}"; do
      for not_cover_penalty in "${not_cover_penalties[@]}"; do
        # Make sure max_cluster_fraction is greater than min_cluster_fraction
        # bc => basic calculator command in Linux. -l allows for floating point calculations
        if (( $(echo "$max_cf > $min_cf" | bc -l) )); then
          # Construct the experiment directory to include parameter settings for organization
          exp_dir_suffix="min${min_cf}_max${max_cf}_overlap${overlap_penalty}_notcover${not_cover_penalty}"
          full_exp_dir="experiments/vicunadata_small/${exp_dir_suffix}"

          # Construct the full command with the current set of parameters
          full_command="$base_command \
            --min_cluster_fraction $min_cf \
            --max_cluster_fraction $max_cf \
            --cluster_overlap_penalty $overlap_penalty \
            --cluster_not_cover_penalty $not_cover_penalty \
            --exp_dir $full_exp_dir \
            --verbose"

          echo "Running Goal-Ex with parameters: $exp_dir_suffix"
          eval "CUDA_VISIBLE_DEVICES=${gpu} $full_command"
          echo "Finished run with parameters: $exp_dir_suffix" 
          echo "--------------------------------"
        fi
      done
    done
  done
done
