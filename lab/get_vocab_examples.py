""" list examples from pickle files in vocab folder"""
import json
import pickle

from pathlib import Path


if __name__ == "__main__":
    vocab_folder = Path("../.cache/vocabs/")
    pkl_list = list(vocab_folder.glob("*.pkl"))
    vocab_example_map = {}

    for pth in pkl_list:
        vocab = pickle.load(pth.open("rb"), encoding="utf8")
        vocab_example_map[pth.stem] = [vocab.itos[i] for i in range(10)]

    json.dump(vocab_example_map, vocab_folder.joinpath("examples2.json").open("w"))
