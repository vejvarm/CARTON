import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from tqdm import tqdm


def categorize_by_num_of_triples(input_dir, output_dir, splits=("test", "train", "val")):
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    total_count = 0
    count_by_num_triples = {s: Counter() for s in splits}
    for json_file in tqdm(input_path.glob('**/*.json')):
        with json_file.open('r', encoding='utf-8') as file:
            data = json.load(file)
            assert len(data) == 2
            user = data[0]
            system = data[1]
            total_count += 1

        # Construct the new path to maintain the subdirectory structure
        name = json_file.name
        relative_path = json_file.relative_to(input_path)
        new_path = output_path.joinpath(relative_path)
        qa = new_path.parent.name
        split_path = new_path.parent.parent
        split = split_path.name

        # check number of triples in the current sample
        num_triples = len(system["active_set"])
        count_by_num_triples[split][num_triples] += 1

        subfolder = f"{num_triples}triples"
        full_new_path = split_path.joinpath(subfolder).joinpath(qa).joinpath(name)

        # Ensure the subdirectories exist in the output directory
        full_new_path.parent.mkdir(parents=True, exist_ok=True)

        # Move the file to the new directory
        shutil.move(str(json_file), str(full_new_path))

    return total_count, count_by_num_triples


def main(args):
    total, num_triples_counter = categorize_by_num_of_triples(args.input, args.output)

    total_counter = Counter()
    [total_counter.update(ctr) for ctr in num_triples_counter.values()]
    num_triples_counter["total"] = total_counter

    print(f"FILES TOTAL: {total}")
    print(f"COUNT by NUM TRIPLES: {num_triples_counter}")

    stat_file_path = Path(args.output).joinpath(args.stat_file)
    json.dump(num_triples_counter, stat_file_path.open("w"), indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Categorize CSQA dataset by number of triples')
    parser.add_argument('--input', required=True, help='Root folder for the source JSON files.')
    parser.add_argument('--output', required=True, help='Root folder for the output JSON files.')
    parser.add_argument('--stat-file', required=False, default="stats.json",
                        help="Name of file for statistics of the process.")

    main(parser.parse_args())
