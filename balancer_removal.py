import os
import shutil
import argparse

def copy_directory(source_dir, destination_dir):

    if os.path.exists(destination_dir):
        shutil.rmtree(destination_dir)

    try:
        shutil.copytree(source_dir, destination_dir)
    except Exception as e:
        print(f"Failed to copy directory: {e}")

def undersample_classes(base_dir):
    # Dictionary to hold the paths of files for each class
    class_files = {}

    # Step 1: Collect all files from each subdirectory
    for subdir in os.listdir(base_dir):
        dir_path = os.path.join(base_dir, subdir)
        if os.path.isdir(dir_path):
            files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
            class_files[subdir] = files

    # Step 2: Find the minimum number of files across classes
    min_count = min(len(files) for files in class_files.values())

    # Step 3: Reduce files in classes with more than min_count files by removing the first excess files
    for class_name, files in class_files.items():
        if len(files) > min_count:
            files_to_remove = files[min_count:]  # Select the excess files starting from min_count
            for file in files_to_remove:
                os.remove(file)  # Remove the file
                print(f"Removed {file}")

    print("Class undersampling complete.")

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, help='Path to balance')
    args = parser.parse_args()
    root_dir = args.base_dir
    output_root_dir = f'{root_dir}-balanced-removal'
    copy_directory(root_dir,output_root_dir)
    undersample_classes(output_root_dir)
