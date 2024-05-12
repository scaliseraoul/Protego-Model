import os
from pydub import AudioSegment
import argparse
import shutil

def split_audio(file_path, output_path, segment_length_ms, overlap_ms):
    audio = AudioSegment.from_file(file_path)
    
    # Total length of the audio in milliseconds
    total_length_ms = len(audio)

    # Calculate potential start points for segments
    start_points = []
    current_start = 0
    while current_start + segment_length_ms <= total_length_ms:
        start_points.append(current_start)
        current_start += segment_length_ms - overlap_ms

    # Ensure the last segment covers the end of the audio
    if total_length_ms >= segment_length_ms and (total_length_ms - start_points[-1] > overlap_ms):
        # Add a start point for the last segment that ensures the last part is also exactly segment_length_ms
        start_points.append(total_length_ms - segment_length_ms)

    # Process all segments
    for i, start_ms in enumerate(start_points):
        end_ms = start_ms + segment_length_ms
        segment = audio[start_ms:end_ms]
        segment_file_name = f"{os.path.splitext(os.path.basename(file_path))[0]}_part_{i}.wav"
        segment_file_path = os.path.join(output_path, segment_file_name)
        segment.export(segment_file_path, format="wav")
        print(f"Exported {segment_file_path}")

def process_directory(root_dir, output_root_dir,segment_length_ms, overlap_ms):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.wav'):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, root_dir)
                output_dir = os.path.join(output_root_dir, relative_path)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                split_audio(file_path, output_dir, segment_length_ms, overlap_ms)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--segment_length', type=int, default=8, help='Segment length to trim')
    parser.add_argument('--change_ambiguous_to', type=str, default='ambiguous', help='Change ambiguous labels to this class')
    parser.add_argument('--overlap_percentage', type=int, default=0, help='Overlap window percentage')

    args = parser.parse_args()

    segment_length = args.segment_length
    overlap_percentage = args.overlap_percentage
    change_ambiguous_to = args.change_ambiguous_to

    segment_length_ms = segment_length * 1000
    overlap_ms = (overlap_percentage/100)*segment_length_ms

    root_dir = f'test-{change_ambiguous_to}-{segment_length}'
    output_root_dir = f'test-{change_ambiguous_to}-{segment_length}-trimmed-{overlap_percentage}'
    if os.path.exists(output_root_dir):
        shutil.rmtree(output_root_dir)
        print(f"Removed existing directory: {output_root_dir}")  
    process_directory(root_dir, output_root_dir,segment_length_ms, overlap_ms)

    ##

    root_dir = f'train-{change_ambiguous_to}-{segment_length}'
    output_root_dir = f'train-{change_ambiguous_to}-{segment_length}-trimmed-{overlap_percentage}'
    if os.path.exists(output_root_dir):
        shutil.rmtree(output_root_dir)
        print(f"Removed existing directory: {output_root_dir}")  
    process_directory(root_dir, output_root_dir,segment_length_ms, overlap_ms)
