import json
from pathlib import Path
from collections import defaultdict

import pandas as pd
from matplotlib import pyplot as plt

ROOT = Path("/media/freya/kubuntu-data/PycharmProjects/CARTON/experiments/inference/v2/")
assert ROOT.exists()

SAVE_FOLDER = ROOT.joinpath("inference_plots")
SAVE_FOLDER.mkdir(exist_ok=True)

SUBTASKS = {"logical_form": "LF",
            "ner": "NER",
            "coref": "COREF",
            "predicate_pointer": "PP",
            "type_pointer": "TP"}

if __name__ == "__main__":
    trains = ["csqa15", "merged15"]
    test_sets = ["csqa", "d2t", "merged"]
    metrics = ["acc", "rec"]  # .json files

    data = defaultdict(dict)
    average_data = []

    for test in test_sets:
        for tr in trains:
            folders = list(ROOT.glob(f"*{tr}*{test}"))
            assert len(folders) == 1

            for metric in metrics:
                fl = folders[0].joinpath(f"{metric}.json")
                metric_data = json.load(fl.open())
                average_data.append({'test_set': test, 'train': tr, 'metric': metric, 'average': metric_data['average']})
                metric_data = {v: metric_data[k] for k, v in SUBTASKS.items()}
                data[metric][tr] = metric_data

        for metric in metrics:
            df = pd.DataFrame(data[metric])

            # Plotting the individual bar chart
            ax = df.plot(kind='bar', figsize=(10, 6))
            ax.set_title(f"{test.upper()} test set | {metric.capitalize()}")
            ax.set_xlabel("Subtask")
            ax.set_ylabel("Values")
            plt.xticks(rotation=45)
            plt.tight_layout()
            df.to_csv(SAVE_FOLDER.joinpath(f"{test}_{metric}.csv"))
            plt.savefig(SAVE_FOLDER.joinpath(f"{test}_{metric}.png"), dpi=300)

    # Convert the list of dictionaries to a DataFrame
    avg_df = pd.DataFrame(average_data)
    avg_df_pivot = avg_df.pivot(index='test_set', columns=['train', 'metric'], values='average')

    # Separate plots for 'acc' and 'rec'
    for metric in metrics:
        avg_df_filtered = avg_df_pivot.xs(metric, level='metric', axis=1)
        avg_df_filtered.plot(kind='bar', figsize=(10, 6))
        plt.title(f"Global Averages | {metric.capitalize()}")
        plt.xlabel("Test Set")
        plt.ylabel("Average Values")
        plt.xticks(rotation=45)
        plt.tight_layout()
        avg_df_filtered.to_csv(SAVE_FOLDER.joinpath(f"average_{metric}.csv"))
        plt.savefig(SAVE_FOLDER.joinpath(f"average_{metric}.png"), dpi=300)
    plt.show()