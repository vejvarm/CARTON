from collections import Counter
from pathlib import Path
from tqdm import tqdm
import shutil
import random
import json
import argparse


def list_files(directory, shuffle=False):
    """List all JSON files starting with 'QA_' in a given directory."""
    files = list(Path(directory).rglob('QA_*.json'))
    if shuffle:
        random.shuffle(files)
    return files


def copy_files(csqa_d2t_files, dest_directory, csqa_d2t_size):
    """Copy percentage of categorized CSQA-D2T files to new directory."""

    total_files = len(csqa_d2t_files)

    for i, file in tqdm(enumerate(csqa_d2t_files), desc="Copying D2T files", leave=True, total=total_files):
        folder_path = dest_directory / file.parent.name
        folder_path.mkdir(parents=True, exist_ok=True)

        # open mapping file in the split root
        with open(dest_directory / "mapping.jsonl", 'a') as f:
            # Copy file to new path
            new_file_path = folder_path / file.name
            shutil.copy(str(file), str(new_file_path))
            # update mapping file
            f.write(json.dumps({str(new_file_path): str(file)}) + '\n')

        if i >= csqa_d2t_size:
            print(f"\tdone! Sampled {i}/{total_files} into `{dest_directory}`.")
            break


def calculate_num_files_to_sample(csqa_size, ratio):
    """Calculate the number of CSQA-D2T samples to include based on the ratio."""
    return int(csqa_size * ratio)


def main(args):
    """ Copy percentage of CSQA-D2T files to `args.target_dir` based on specified `args.percentage`. """
    csqa_d2t_dir = Path(args.csqa_d2t_dir)
    target_dir = Path(args.target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    percentage = float(args.percentage)
    assert percentage > 0.
    splits = tuple(args.splits)

    for subset in splits:
        print(f"\n\nsplit: {subset.upper()}")
        csqa_d2t_files = list_files(csqa_d2t_dir / subset, shuffle=True)

        # calculate num samples from D2T for enrichment from percentage
        csqa_d2t_size = calculate_num_files_to_sample(len(csqa_d2t_files), percentage)
        # print(f"path examples: {csqa_d2t_files[:10]}")
        # print(f"new size: {csqa_d2t_size}/{len(csqa_d2t_files)}")
        # continue # DEBUG

        # copy files
        copy_files(csqa_d2t_files, target_dir / subset, csqa_d2t_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge CSQA and CSQA-D2T datasets.")

    # Required arguments
    parser.add_argument('--csqa-d2t-dir', type=str, help='Path to the categorized CSQA-D2T dataset directory.')
    parser.add_argument('--target-dir', type=str, help='Path to the directory where the sampled dataset will be stored.')

    # Optional arguments
    parser.add_argument('--percentage', type=float, default=0.2, help='Ratio of CSQA-D2T samples to include (0. to 1.).')
    parser.add_argument('--splits', nargs='+', default=["test", "train", "val"],
                        choices=["test", "train", "val"], help='Dataset splits to process.')

    # Call the main function with parsed arguments
    main(parser.parse_args())

# if __name__ == "__main__":
#     csqa_dir = "/media/freya/kubuntu-data/PycharmProjects/CARTON/data/final/csqa"  # required
#     csqa_d2t_dir = "/media/freya/kubuntu-data/datasets/CARTONNER/csqa-categorized"  # required
#     target_dir = "/media/freya/kubuntu-data/datasets/CARTONNER/csqa-merged"  # required
#     ratio = 0.2  # float from 0 to 1
#     splits = ["test", "train", "val"]  # list, with default being all three options
#
#     main(csqa_dir, csqa_d2t_dir, target_dir, ratio, splits)
