""" list examples from pickle files in vocab folder"""
import json
import pickle

from pathlib import Path


if __name__ == "__main__":
    vocab_folder = Path("../.cache/vocabs/")
    pkl_list = list(vocab_folder.glob("*.pkl"))
    stats = {}
    stats['sizes'] = {}
    stats['examples'] = {}

    for pth in pkl_list:
        vocab = pickle.load(pth.open("rb"), encoding="utf8")
        stats['sizes'][pth.stem] = len(vocab)
        stats['examples'][pth.stem] = [vocab.itos[i] for i in range(10)]

    json.dump(stats, vocab_folder.joinpath("stats.json").open("w"), indent=4)
    print("done.")
