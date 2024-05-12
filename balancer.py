import os
import shutil
from random import choice

def balance_classes(base_dir):
    # Dictionary to hold the count of files for each class
    class_counts = {}
    class_files = {}

    # Step 1: Loop through each subfolder and count files
    for subdir in os.listdir(base_dir):
        dir_path = os.path.join(base_dir, subdir)
        if os.path.isdir(dir_path):
            files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
            class_counts[subdir] = len(files)
            class_files[subdir] = files

    # Step 2: Find the class with the maximum number of files
    max_count = max(class_counts.values())

    # Step 3: Duplicate files in underrepresented classes
    for class_name, files in class_files.items():
        current_count = class_counts[class_name]
        if current_count < max_count:
            dir_path = os.path.join(base_dir, class_name)
            while class_counts[class_name] < max_count:
                file_to_duplicate = choice(files)
                original_file_path = os.path.join(dir_path, file_to_duplicate)
                new_file_path = os.path.join(dir_path, f"copy_{class_counts[class_name]}_{file_to_duplicate}")
                shutil.copyfile(original_file_path, new_file_path)
                class_counts[class_name] += 1

    # Output the final counts for verification
    print("Final counts for each class:", class_counts)

# Example usage
if __name__ == "__main__":
    base_dir = 'test_aggression-trimmed-balanced-removal-8-0'  # Change this to the path of your base directory
    balance_classes(base_dir)
