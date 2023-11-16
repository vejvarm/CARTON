import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from tqdm import tqdm


def categorize_by_num_of_triples(input_dir, splits=("test", "train", "val")):
    input_path = Path(input_dir)

    total_count = 0
    count_by_num_triples = {s: Counter() for s in splits}
    count_convo_len = Counter()
    for json_file in tqdm(input_path.glob('**/QA_*.json')):
        with json_file.open('r', encoding='utf-8') as file:
            data = json.load(file)
            count_convo_len[len(data)//2] += 1

        split_path = json_file.parent.parent
        split = split_path.name

        for i in range(0, len(data), 2):
            # user = data[i]
            system = data[i+1]
            total_count += 1
            # Construct the new path to maintain the subdirectory structure

            # check number of triples in the current sample
            num_triples = len(system["active_set"])
            count_by_num_triples[split][num_triples] += 1

    return total_count, count_by_num_triples, count_convo_len


def main(args):
    total, num_triples_counter, convo_len_counter = categorize_by_num_of_triples(args.input)

    stats = {**num_triples_counter}

    total_counter = Counter()
    [total_counter.update(ctr) for ctr in num_triples_counter.values()]
    stats["total"] = total_counter
    stats["convo_len"] = convo_len_counter

    print(f"FILES TOTAL: {total}")
    print(f"STATS: {stats}")

    stat_file_path = Path(args.input).joinpath(args.stat_file)
    json.dump(stats, stat_file_path.open("w"), indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Categorize CSQA dataset by number of triples')
    parser.add_argument('--input', required=True, help='Root folder for the source JSON files.')
    parser.add_argument('--stat-file', required=False, default="stats.json",
                        help="Name of file for statistics of the process.")

    main(parser.parse_args())
