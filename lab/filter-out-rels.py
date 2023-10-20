import json
import argparse
from pathlib import Path
import shutil
from tqdm import tqdm


def filter_and_move_files(input_dir, output_dir, filtered_rels):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    filtered_rels_set = set(filtered_rels)

    total_count = 0
    omitted_count = 0
    for json_file in tqdm(input_path.glob('**/*.json')):
        with json_file.open('r', encoding='utf-8') as file:
            data = json.load(file)

        total_count += 1

        # Check if any of the filtered relations are in any of the JSON objects
        if any(filtered_rel in data[0]['relations'] for filtered_rel in filtered_rels_set):
            omitted_count += 1
            continue  # Skip this file as it contains one of the filtered relations

        # Construct the new path to maintain the subdirectory structure
        relative_path = json_file.relative_to(input_path)
        new_path = output_path.joinpath(relative_path)

        # Ensure the subdirectories exist in the output directory
        new_path.parent.mkdir(parents=True, exist_ok=True)

        # Move the file to the new directory
        shutil.move(str(json_file), str(new_path))

    return total_count, omitted_count


def main(args):
    total, omitted = filter_and_move_files(args.input, args.output, args.filtered_rels)

    print(f"FILES TOTAL: {total}")
    print(f"FILES MOVED: {total - omitted}")
    print(f"FILES OMITTED: {omitted}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Filter and move JSON files based on relations field.')
    parser.add_argument('--input', required=True, help='Root folder for the source JSON files.')
    parser.add_argument('--output', required=True, help='Root folder for the output JSON files.')
    parser.add_argument('--filtered-rels', nargs='+', default=["P105", "P171"], help='List of relations which should be omitted (i.e. "P105 P171").')

    main(parser.parse_args())
