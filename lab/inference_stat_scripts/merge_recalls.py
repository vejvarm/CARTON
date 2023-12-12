import json
from pathlib import Path
from os import remove


def load_data(pth: Path, ignore=("average", )) -> (dict, dict):
    data = json.load(pth.open())
    ignored_vals = {k: data.pop(k) for k in ignore if k in data.keys()}
    return data, ignored_vals


if __name__ == "__main__":
    source_folders = Path("/media/freya/kubuntu-data/PycharmProjects/CARTON/experiments/inference/v2").glob("*")

    for fldr in source_folders:
        if not fldr.is_dir():
            continue
        rec, _ = load_data(fldr.joinpath("rec.json"))
        rec_ner, _ = load_data(fldr.joinpath("rec-ner.json"))

        rec.update(rec_ner)

        rec["average"] = sum(rec.values()) / len(rec)
        json.dump(rec, fldr.joinpath("rec.json").open("w"), indent=4)
        remove(fldr.joinpath("rec-ner.json"))
    print("done")
