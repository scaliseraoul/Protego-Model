#!/bin/bash

# Define arrays for variant and overlap_percentage
variants=("neutral")
overlap_percentages=(0 75)
lengths=(4)

# Loop over all combinations of variant, overlap_percentage, and length
for length in "${lengths[@]}"; do
    for variant in "${variants[@]}"; do
        echo "Processing with variant: $variant, and segment length: $length"
        python train.py --train_dir "train-${variant}-${length}" --test_dir "test-${variant}-${length}"
        for overlap in "${overlap_percentages[@]}"; do
            echo "Processing with overlap: $overlap"
            python train.py --train_dir "train-${variant}-${length}-trimmed-${overlap}" --test_dir "test-${variant}-${length}-trimmed-${overlap}"
            python train.py --train_dir "train-${variant}-${length}-trimmed-${overlap}-balanced-removal" --test_dir "test-${variant}-${length}-trimmed-${overlap}-balanced-removal"
            echo "Completed processing for variant: $variant, overlap: $overlap%, and segment length: $length"
        done
    done
done

echo "All processing complete."
