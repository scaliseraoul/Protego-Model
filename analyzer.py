import json
import os
from urllib.parse import unquote

# Load JSON Data
def load_annotations(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    return data

# Extract and Save Audio Segments
def analyze(data):

    for item in data:
        video_local_path = unquote(item['data']['video_url'].split('?d=')[1])  # Extracting the file path
        video_path = os.path.join('/Users/raoul/', video_local_path)  # Adjust base path as necessary
        annotations = item['annotations'][0]['result']
        total_length = annotations[0]['original_length']

        if not os.path.exists(video_path):
            print(f"{video_local_path} - Error")
        else:
            print(f"{video_local_path} - {total_length} Ok")


# Main function to run the script
def main():
    json_file = 'annotations-train.json'

    annotations = load_annotations(json_file)
    analyze(annotations)

if __name__ == "__main__":
    main()
