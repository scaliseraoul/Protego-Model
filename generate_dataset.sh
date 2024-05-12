#!/bin/bash

# Define arrays for variant and overlap_percentage
variants=("neutral")
overlap_percentages=(0 75)
lengths=(4)

# Loop over all combinations of variant, overlap_percentage, and length
for length in "${lengths[@]}"; do
    for variant in "${variants[@]}"; do
        echo "Processing with variant: $variant, and segment length: $length"
        python segment_generator.py --min_length "$length" --change_ambiguous_to "$variant"
        for overlap in "${overlap_percentages[@]}"; do
            echo "Processing with overlap: $overlap"
            python split_generator.py --segment_length "$length" --change_ambiguous_to "$variant" --overlap_percentage "$overlap"
            python balancer_removal.py --base_dir "train-${variant}-${length}-trimmed-${overlap}"
            python balancer_removal.py --base_dir "test-${variant}-${length}-trimmed-${overlap}"
            echo "Completed processing for variant: $variant, overlap: $overlap%, and segment length: $length"
        done
    done
done

echo "All processing complete."
