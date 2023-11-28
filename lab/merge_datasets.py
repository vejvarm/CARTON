from pathlib import Path
from tqdm import tqdm
import shutil
import random
import json
import argparse

def list_files(directory, shuffle=False):
    """List all JSON files in a given directory."""
    files = list(Path(directory).rglob('QA_*.json'))
    if shuffle:
        random.shuffle(files)
    return files


def copy_files(csqa_files, csqa_d2t_files, dest_directory, csqa_d2t_size):
    """Copy and evenly distribute CSQA-D2T files across CSQA subfolders, starting after the highest existing file number."""
    csqa_folders = {}
    for file_path in csqa_files:
        folder = file_path.parent.name
        csqa_folders.setdefault(folder, []).append(file_path)

    num_csqa_d2t_per_folder = round(csqa_d2t_size / len(csqa_folders))

    csqa_d2t_iter = iter(csqa_d2t_files)
    with tqdm(desc="Copying D2T files", total=csqa_d2t_size, leave=True) as pbar:
        for fldr_idx, (folder, files) in enumerate(csqa_folders.items()):
            folder_path = dest_directory / folder
            folder_path.mkdir(parents=True, exist_ok=True)

            # Find the maximum 'y' in QA_y.json
            max_y = max(int(file.name.split('_')[1].split('.')[0]) for file in files)

            # open mapping file in the split root
            with open(dest_directory / "mapping.jsonl", 'a') as f:
                pbar.set_postfix({'folder': f"{folder_path.name} ({fldr_idx+1}/{len(csqa_folders)})"})
                # Copy CSQA files
                for file_path in files:
                    new_file_path = folder_path / file_path.name
                    shutil.copy(str(file_path), str(new_file_path))
                    # update mapping file
                    f.write(json.dumps({str(new_file_path): str(file_path)}) + '\n')

                # Copy and rename CSQA-D2T files
                for i in range(num_csqa_d2t_per_folder):
                    try:
                        new_file_name = f"QA_{max_y + 1 + i}.json"
                        csqa_d2t_file = next(csqa_d2t_iter)
                        new_file_path = folder_path / new_file_name
                        shutil.copy(str(csqa_d2t_file), str(new_file_path))
                        # update mapping file
                        f.write(json.dumps({str(new_file_path): str(csqa_d2t_file)}))
                        pbar.update(1)
                    except StopIteration:
                        print("No more CSQA-D2T files to distribute.")
                        break

        # copy remaining files to leftovers folder
        folder_path = dest_directory / "QA_-1"
        i = 0
        pbar.set_postfix({'folder': f"{folder_path.name} ({len(csqa_folders)+1}/{len(csqa_folders)})"})
        print(f"total before leftover: {pbar.n}")
        while True:
            try:
                new_file_name = f"QA_{i}.json"
                csqa_d2t_file = next(csqa_d2t_iter)
                new_file_path = folder_path / new_file_name
                if i == 0:
                    folder_path.mkdir(exist_ok=True, parents=True)
                shutil.copy(str(csqa_d2t_file), str(new_file_path))
                # update mapping file
                f.write(json.dumps({str(new_file_path): str(csqa_d2t_file)}))
                i += 1
                pbar.update(1)
            except StopIteration:
                print("No more CSQA-D2T files to distribute.")
                break


def calculate_csqa_d2t_size(csqa_size, ratio):
    """Calculate the number of CSQA-D2T samples to include based on the ratio."""
    return int(csqa_size * ratio)


def main(args):
    """ Copy all CSQA files and add part of CSQA-D2T files based on the specified ratio. """
    csqa_dir = Path(args.csqa_dir)
    csqa_d2t_dir = Path(args.csqa_d2t_dir)
    merged_dir = Path(args.merged_dir)
    merged_dir.mkdir(parents=True, exist_ok=True)

    ratio = float(args.ratio)
    assert ratio > 0. or ratio == -1
    splits = tuple(args.splits)

    # load stats
    try:
        csqa_stats = json.load(csqa_dir.joinpath("stats.json").open())
    except FileNotFoundError:
        print(f"Please run count_triples.py on {csqa_dir} first.")
        exit()

    for subset in splits:
        print(f"\n\nsplit: {subset.upper()}")
        csqa_files = list_files(csqa_dir / subset)
        csqa_d2t_files = list_files(csqa_d2t_dir / subset, shuffle=True)

        # calculate num samples from D2T for enrichment from ratio
        csqa_total_turns = sum(csqa_stats[subset].values())  # equivalent to Counter(csqa_stats[subset]).total()
        if ratio == -1:
            csqa_d2t_size = len(csqa_d2t_files)
        else:
            csqa_d2t_size = calculate_csqa_d2t_size(csqa_total_turns, ratio)

        # copy files
        copy_files(csqa_files, csqa_d2t_files, merged_dir / subset, csqa_d2t_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge CSQA and CSQA-D2T datasets.")

    # Required arguments
    parser.add_argument('--csqa-dir', type=str, help='Path to the CSQA dataset directory.')
    parser.add_argument('--csqa-d2t-dir', type=str, help='Path to the CSQA-D2T dataset directory.')
    parser.add_argument('--merged-dir', type=str, help='Path to the directory where the merged dataset will be stored.')

    # Optional arguments
    parser.add_argument('--ratio', type=float, default=0.2, help='Ratio of CSQA-D2T samples to include (0. to 1.). If -1: merge full datasets together!')
    parser.add_argument('--splits', nargs='+', default=["test", "train", "val"],
                        choices=["test", "train", "val"], help='Dataset splits to process.')

    # Call the main function with parsed arguments
    main(parser.parse_args())

# if __name__ == "__main__":
#     csqa_dir = "/media/freya/kubuntu-data/PycharmProjects/CARTON/data/final/csqa"  # required
#     csqa_d2t_dir = "/media/freya/kubuntu-data/datasets/CARTONNER/csqa-categorized"  # required
#     merged_dir = "/media/freya/kubuntu-data/datasets/CARTONNER/csqa-merged"  # required
#     ratio = 0.2  # float from 0 to 1
#     splits = ["test", "train", "val"]  # list, with default being all three options
#
#     main(csqa_dir, csqa_d2t_dir, merged_dir, ratio, splits)
