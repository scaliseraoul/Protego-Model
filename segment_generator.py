import json
import subprocess
import os
from urllib.parse import unquote
import argparse
import shutil

# Load JSON Data
def load_annotations(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    return data

# Ensure subfolders for each class exist
def ensure_class_folders(output_dir, classes):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"Removed existing directory: {output_dir}")
    for cls in classes:
        class_dir = os.path.join(output_dir, cls)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

# Extract and Save Audio Segments
def extract_audio(data, output_dir, classes,min_length,change_ambiguous_to):
    ensure_class_folders(output_dir, classes)

    min_length = min_length/1000
    for item in data:
        video_path = item['data']['video_url'].split('?d=')[1]  # Extracting the file path
        video_path = os.path.join('/Users/raoul/', video_path)  # Adjust base path as necessary
        video_path = unquote(video_path)
        video_id = item["id"]

        if not os.path.exists(video_path):
            print(f"Error: The file {video_path} does not exist.")
            continue  

        print(f"--- Processing {video_path}")
        annotations = item['annotations'][0]['result']
        
        # Sort annotations by start time
        annotations.sort(key=lambda x: x['value']['start'])
        
        last_end = 0  # Track the end of the last processed segment
        total_length = annotations[0]['original_length']

        # Handle initial gap if it exists
        initial_gap_duration = annotations[0]['value']['start']
        if initial_gap_duration > 0 and initial_gap_duration >= min_length:
            save_segment(video_path, video_id, 0, annotations[0]['value']['start'], 'neutral', output_dir)
            last_end = initial_gap_duration
        
        for annotation in annotations:
            start_time = annotation['value']['start']
            end_time = annotation['value']['end']
            label = annotation['value']['labels'][0]
            segment_duration = end_time - start_time

            # experiment with 2 classes ()
            if label == 'ambiguous':
                label = change_ambiguous_to

            # Skip unwanted segments or too short segments
            if label == 'unwanted' or segment_duration < min_length:
                last_end = end_time
                continue

            # Handle gap as 'neutral' if it exists and it's longer than 0.2 seconds
            if last_end < start_time and (start_time - last_end) >= min_length:
                save_segment(video_path, video_id, last_end, start_time, 'neutral', output_dir)
            
            # Save segment with corresponding label
            save_segment(video_path, video_id, start_time, end_time, label, output_dir)
            last_end = end_time
        
        # Handle the last gap to the end of the video if it's longer than 0.2 seconds
        if last_end < total_length and (total_length - last_end) >= min_length:
            save_segment(video_path, video_id, last_end, total_length, 'neutral', output_dir)

# Function to save audio segment
def save_segment(video_path, video_id, start_time, end_time, label, output_dir):
    output_filename = f"{video_id}_{start_time}_{end_time}.wav"
    output_path = os.path.join(output_dir, label, output_filename)
    command = [
        'ffmpeg', '-i', video_path,
        '-ss', str(start_time), '-to', str(end_time),
        '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', output_path
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print("FFmpeg command failed with return code:", result.returncode)
    else:
        print(f"Extracted {output_path}")

# Main function to run the script
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--min_length', type=int, default=8, help='Minimum segment length to process')
    parser.add_argument('--change_ambiguous_to', type=str, default='ambiguous', help='Change ambiguous labels to this class')

    args = parser.parse_args()

    min_length = args.min_length
    change_ambiguous_to = args.change_ambiguous_to

    classes = ['neutral', 'ambiguous', 'aggression']

    if change_ambiguous_to != 'ambiguous':
        classes = ['neutral', 'aggression']

    train_annotation = 'annotations-train.json'
    train_dir = f"train-{change_ambiguous_to}-{min_length}"
    test_annotation = 'annotations-test.json'
    test_dir = f"test-{change_ambiguous_to}-{min_length}" 
    
    train_annotation = load_annotations(train_annotation)
    test_annotation = load_annotations(test_annotation)
    
    extract_audio(train_annotation, train_dir, classes,min_length,change_ambiguous_to)
    extract_audio(test_annotation, test_dir, classes,min_length,change_ambiguous_to)

if __name__ == "__main__":
    main()
