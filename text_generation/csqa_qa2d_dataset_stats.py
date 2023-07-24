import sys
import csv
import json
import pathlib
from collections import Counter
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from constants import ROOT_PATH


DEFAULT_KEY_TO_LABEL_MAP = {
    "total_samples": "Total Samples",
    "total_unique_entities": "Total unique Entities",
    "all_entities_max": "Max Entities per Sample",
    "all_entities_mean_per_sample": "Mean Entities per Sample",
    "all_entities_mean_repetition": "Mean Entity Repetition",
    "total_unique_relations": "Total unique Relations",
    "relations_max": "Max Relations per Sample",
    "relations_mean_per_sample": "Mean Relations per Sample",
    "relations_mean_repetition": "Mean Relation Repetition",
    "active_set_max_len": "Max number of active sets",
    "active_set_mean_len_per_sample": "Mean Num of active sets",
}


def generate_stats(dataset_folder: pathlib.Path, splits: list[str], stats_filename="stats.json"):
    for split in splits:
        print(f"SPLIT: {split}")
        json_file_paths = list(dataset_folder.joinpath(split).glob("**/QA_*.json"))
        total_samples = 0
        total_active_set_n = 0
        all_entities_max = 0
        relations_max = 0
        active_set_max_len = 0
        active_set_len_counter = Counter()
        description_counter = Counter()
        entity_counter = Counter()
        relation_counter = Counter()

        for json_file_path in tqdm(json_file_paths):
            with json_file_path.open() as json_file:
                json_file_data = json.load(json_file)  # list
                n_samples = len(json_file_data) // 2
                assert float(n_samples) == len(json_file_data) / 2
                total_samples += n_samples

                for i in range(n_samples):
                    user = json_file_data[2 * i]
                    system = json_file_data[2 * i + 1]
                    all_entities_max = max(all_entities_max, len(user["entities_in_utterance"]))
                    relations_max = max(relations_max, len(user["relations"]))
                    active_set = system["active_set"]
                    active_set_len = len(active_set)
                    active_set_max_len = max(active_set_max_len, active_set_len)
                    total_active_set_n += active_set_len
                    active_set_len_counter[str(active_set_len)] += 1
                    if "description" in user.keys():
                        description_counter[user["description"]] += 1
                    else:
                        description_counter["NA"] += 1
                    entity_counter.update(user["entities_in_utterance"])
                    relation_counter.update(user["relations"])

        stats = {
            "total_samples": total_samples,
            "total_unique_entities": len(entity_counter.keys()),
            "all_entities_max": all_entities_max,
            "all_entities_mean_per_sample": entity_counter.total()/total_samples,
            "all_entities_mean_repetition": entity_counter.total()/len(entity_counter.keys()),
            "total_unique_relations": len(relation_counter.keys()),
            "relations_max": relations_max,
            "relations_mean_per_sample": relation_counter.total()/total_samples,
            "relations_mean_repetition": relation_counter.total()/len(relation_counter.keys()),
            "active_set_max_len": active_set_max_len,
            "active_set_mean_len_per_sample": sum(int(l)*count for l, count in active_set_len_counter.items())/total_samples,
            "active_set_len_counter": dict(active_set_len_counter),
            "description_counter": dict(description_counter),
            "entity_counter": dict(entity_counter),
            "relation_counter": dict(relation_counter),
        }

        with dataset_folder.joinpath(split).joinpath(stats_filename).open("w") as stats_file:
            json.dump(stats, stats_file, indent=4)


def load_stats(dataset_folder: pathlib.Path, splits: list[str], stats_filename="stats.json"):
    stats = dict()
    for split in splits:
        stats[split] = json.load(dataset_folder.joinpath(split).joinpath(stats_filename).open())

    return stats


def plot_histogram(counter: Counter, title='Histogram', xlabel='Entities', ylabel='Counts', sort=False):
    """
    Plots a histogram using a Counter object.

    :param counter: A Counter object containing the counts of entities.
    :param title: The title of the histogram (default: 'Histogram').
    :param xlabel: The label for the x-axis (default: 'Entities').
    :param ylabel: The label for the y-axis (default: 'Counts').
    :param sort: bool flag, wherther to sort by counter key values (only works for int parseable vals)
    """
    # # Extract the entities and their counts from the Counter object
    # entities, counts = zip(*counter.items())

    # Plot the histogram
    if sort:
        plt.bar(*list(zip(*sorted(counter.items(), key=lambda x: int(x[0])))))
    else:
        plt.bar(counter.keys(), counter.values())

    # Set the labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)


def plot_grouped_histogram(counters: list[Counter], labels: list[str],
                           normalization_factors: Optional[list[float]] = None,
                           title='Grouped Histogram', xlabel='Categories', ylabel='Counts', sort=False):
    """
    Plots a grouped bar chart using a list of Counter objects.

    :param counters: A list of Counter objects containing the counts of entities.
    :param labels: A list of labels for each group (corresponding to the Counter objects).
    :param normalization_factors: A list of normalization factors for each Counter object in the 'counters' list.
        Each count value will be divided by the corresponding normalization factor.
        The length of this list should be the same as the length of the 'counters' list.
    :param title: The title of the histogram (default: 'Grouped Histogram').
    :param xlabel: The label for the x-axis (default: 'Categories').
    :param ylabel: The label for the y-axis (default: 'Counts').
    :param sort: bool flag, whether to sort by counter key values (only works for int parseable vals)
    """

    num_counters = len(counters)
    counter_keys = sorted(set.union(*(set(c.keys()) for c in counters)))

    if sort:
        counter_keys = sorted(counter_keys, key=lambda x: int(x))

    # Set the width of a bar
    bar_width = 0.8 / num_counters

    # Set the position of bars on the x-axis
    positions = np.arange(len(counter_keys))

    for i, (counter, label) in enumerate(zip(counters, labels)):
        if normalization_factors is not None:
            values = [counter.get(k, 0) / normalization_factors[i] for k in counter_keys]
        else:
            values = [counter.get(k, 0) for k in counter_keys]
        plt.bar(positions + i * bar_width, values, width=bar_width, label=label)

    # Set the labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(positions + bar_width * (num_counters - 1) / 2, counter_keys)

    # Adding the legend
    plt.legend()


def write_csv_file(stats, splits: list, output_file: Optional[str] = None, key_to_label_map: Optional[dict] = None):
    fieldnames = ["Stat"] + splits

    if key_to_label_map is None:
        key_to_label_map = DEFAULT_KEY_TO_LABEL_MAP

    with open(output_file, 'w', newline='') if output_file else sys.stdout as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for stat_key, label in key_to_label_map.items():
            row = {"Stat": label}
            for split in splits:
                row[split] = stats[split][stat_key]
            writer.writerow(row)


def main():
    dataset_folder = pathlib.Path("/media/freya/kubuntu-data/datasets/csqa/CARTON/data/original_simple_direct_csqa/")
    results_folder = dataset_folder.joinpath("stats")
    splits = ["val", "test", "train"]
    figsize = (8, 4.5)
    results_folder.mkdir(parents=True, exist_ok=True)
    dataset_label = "CSQA-QA2D"

    # generate_stats(dataset_folder, splits)
    stats = load_stats(dataset_folder, splits)

    # write stats to csv file:
    write_csv_file(stats, splits, output_file=results_folder.joinpath("all_splits_stats.csv"))

    for split in splits:
        print(
            f"all_entities_max: {stats[split]['all_entities_max']}\tall_entities_mean_per_sample: {stats[split]['all_entities_mean_per_sample']}\tall_entities_mean_repetition: {stats[split]['all_entities_mean_repetition']}")
        print(
            f"relations_max: {stats[split]['relations_max']}\trelations_mean_per_sample: {stats[split]['relations_mean_per_sample']}\trelations_mean_repetition: {stats[split]['relations_mean_repetition']}")
        print(stats[split]["all_entities_max"])

        # active_set_len_counter plot
        active_set_len_counter = Counter(stats[split]["active_set_len_counter"])
        fig, ax = plt.subplots(figsize=figsize)
        plot_histogram(active_set_len_counter,
                       title=f"Number of samples grouped by RDF count in  {dataset_label} for {split} split",
                       xlabel="number of rdf entries", sort=True)
        ax.set_yscale('log')
        plt.tight_layout()
        fig.savefig(results_folder.joinpath(f"hist_active_set_len_counter_{split}.png"), dpi=300)

        # description_counter plot
        description_counter = Counter(stats[split]["description_counter"])
        fig, ax = plt.subplots(figsize=figsize)
        plot_histogram(description_counter,
                       title=f"Number of samples grouped by insert categories in {dataset_label} for {split} split",
                       xlabel="sample category", sort=False)
        ax.set_yscale('log')
        plt.tight_layout()
        fig.savefig(results_folder.joinpath(f"hist_description_counter_{split}.png"), dpi=300)

    normalization_factors = [stats[split]["total_samples"] for split in splits]

    norm_options = {"Absolute": None,
                    "Sample normalized": normalization_factors}

    colors = {'dev': '#1f77b4', 'test': '#ff7f0e', 'train': '#2ca02c'}
    bar_colors = list(colors.values())

    for lab, norm_factors in norm_options.items():
        # active_set_len_counter plot
        active_set_len_counters = [Counter(stats[split]["active_set_len_counter"]) for split in splits]
        fig, ax = plt.subplots(figsize=figsize)
        plot_grouped_histogram(active_set_len_counters, splits, norm_factors,
                               title=f'{lab} number of samples grouped by RDF count in {dataset_label}',
                               xlabel=f'number of rdf entries in sample', ylabel='number of samples', sort=True)
        ax.set_yscale('log')
        plt.tight_layout()
        fig.savefig(results_folder.joinpath(f"hist_active_set_len_counter_grouped_{lab}.png"), dpi=300)

        # description_counter plot
        description_counters = [Counter(stats[split]["description_counter"]) for split in splits]
        fig, ax = plt.subplots(figsize=figsize)
        plot_grouped_histogram(description_counters, splits, norm_factors,
                               title=f'{lab} number of samples grouped by insert categories in {dataset_label}',
                               xlabel='sample category', ylabel='number of samples', sort=False)
        ax.set_yscale('log')
        plt.tight_layout()
        fig.savefig(results_folder.joinpath(f"hist_description_counter_grouped_{lab}.png"), dpi=300)

        # total_unique_entities plot
        total_unique_entities = [stats[split]["total_unique_entities"] for split in splits]
        if norm_factors is not None:
            total_unique_entities = np.divide(total_unique_entities, norm_factors)
        fig, ax = plt.subplots(figsize=figsize)
        plt.bar(splits, total_unique_entities, color=bar_colors)
        plt.xlabel('Splits')
        plt.ylabel(f'{lab.lower()} count of entities')
        plt.title(f'{lab} unique entities in {dataset_label}')
        plt.tight_layout()
        fig.savefig(results_folder.joinpath(f"bar_total_unique_entities_grouped_{lab}.png"), dpi=300)

        # total_unique_relations plot
        total_unique_relations = [stats[split]["total_unique_relations"] for split in splits]
        if norm_factors is not None:
            total_unique_relations = np.divide(total_unique_relations, norm_factors)
        fig, ax = plt.subplots(figsize=figsize)
        plt.bar(splits, total_unique_relations, color=bar_colors)
        plt.xlabel('Splits')
        plt.ylabel(f'{lab.lower()} count of relations')
        plt.title(f'{lab} unique relations in {dataset_label}')
        plt.tight_layout()
        fig.savefig(results_folder.joinpath(f"bar_total_unique_relations_grouped_{lab}.png"), dpi=300)

    plt.show()


if __name__ == "__main__":
    main()