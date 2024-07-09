youtube-dl -f 'best[ext=mp4]' -a file_to_download.txt

find . -type d -exec sh -c "echo '{}: '; ls '{}' | wc -l" \;

python train.py --train_dir 'train-neutral-4-trimmed-0' --test_dir 'train-neutral-4-trimmed-0'
python train2.py --train_dir 'train-neutral-4-trimmed-0' --test_dir 'train-neutral-4-trimmed-0'
python trainyamnet.py --train_dir 'train-neutral-4-trimmed-0' --test_dir 'train-neutral-4-trimmed-0'