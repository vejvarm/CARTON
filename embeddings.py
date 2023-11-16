import pathlib

import jsonlines
import numpy
# import scipy.stats
import scipy
import torch
import flair
from flair.data import Sentence
from flair.embeddings import BertEmbeddings, DocumentPoolEmbeddings
import random
import json
from constants import ROOT_PATH

# Set device
torch.cuda.set_device(0)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
flair.device = DEVICE


class EmbeddingGenerator:
    def __init__(self, emb_file_path: pathlib.Path = ROOT_PATH.joinpath("knowledge_graph").joinpath("entity_embeddings.jsonl")):
        """ to use call: (id, label, emb) = EmbeddingGenerator().add_entry(label)
        :param emb_file_path: path to jsonl file with entity ids and their embeddings (will make new if doesn't exist)
        """
        # Load BERT model
        self.bert = DocumentPoolEmbeddings([BertEmbeddings('bert-base-uncased', layers='-1')])
        self.emb_file_path = pathlib.Path(emb_file_path)

        # Load embedding dict from existing file or just make a new empty jsonl file
        self.id_emb, self.lab_id = self._load_database(self.emb_file_path)


    def check_existance(self, label):
        if label in self.lab_id.keys():
            return True
        else:
            return False

    def generate_embedding(self, label):
        # Create a Sentence object
        flair_sentence = Sentence(label)

        # Use BERT to generate the embedding for the sentence
        self.bert.embed(flair_sentence)
        embedding = flair_sentence.embedding.detach().cpu().tolist()

        # Generate a random ID
        id = 'Q' + ''.join([str(random.randint(0, 9)) for _ in range(10)])
        # TODO: use wikidata id instead of random?

        # Create and return a tuple
        return id, embedding

    @staticmethod
    def _load_database(db_file_path: pathlib.Path):
        """ # !NOTE: in production we would use database to store embeddings, here we use jsons """
        id_emb = {}
        lab_id = {}
        if db_file_path.exists():
            with jsonlines.Reader(db_file_path.open("r")) as reader:
                for ln in reader.iter():
                    id_emb.update({ln[0]: ln[2]})
                    lab_id.update({ln[1]: ln[0]})
        else:
            db_file_path.parent.mkdir(exist_ok=True)

        return id_emb, lab_id

    @staticmethod
    def _update_database(id_lab_emb_tupl: tuple[str, str, numpy.ndarray], db_file_path: pathlib.Path):
        """ # !NOTE: in production we would use database to store these embeddings, here we use jsons """
        # we do not check for existence!
        with db_file_path.open('a') as outfile:
            outfile.write(f"{json.dumps(id_lab_emb_tupl)}\n")

    def add_entry(self, label):

        # if it exists, return the existing entry
        if self.check_existance(label):
            id = self.lab_id[label]
            tupl = (id, label, self.id_emb[id])
            print(f"EXISTING ENTRY: {tupl[0]}: {tupl[1]}")  # DEBUG
            return tupl

        # if it doesn't exist generate new entry
        id, embedding = self.generate_embedding(label)

        # update dictionaries
        self.id_emb[id] = embedding
        self.lab_id[label] = id

        # add entry to database
        tupl = (id, label, embedding)
        self._update_database(tupl, self.emb_file_path)

        print(f"NEW ENTRY: {tupl[0]}: {tupl[1]}")  # DEBUG
        return tupl


def test_embedding_generator():
    # Create an EmbeddingGenerator object
    embedding_generator = EmbeddingGenerator()

    # Generate and print embeddings for each label
    embedding_tupl = embedding_generator.add_entry("apple")
    id_apple, emb_apple = embedding_generator.generate_embedding("apple")
    embedding_tupl_different = embedding_generator.add_entry("apples")

    print(f"{embedding_tupl[2] == emb_apple} (T) | {embedding_tupl[2] == embedding_tupl_different[2]} (F)")
    print(f"{embedding_tupl[1]}: {embedding_tupl[2]}")
    print(f"{id_apple}: {emb_apple}")
    print(f"{embedding_tupl_different[1]}: {embedding_tupl_different[2]}")


def test_similarity():
    # Create an EmbeddingGenerator object
    embedding_generator = EmbeddingGenerator()

    # Generate and print embeddings for each label
    id_apple, _, emb_apple = embedding_generator.add_entry("apple")
    id_apples, emb_apples = embedding_generator.generate_embedding("apples")
    id_adamsapple, _, emb_adamsapple =  embedding_generator.add_entry("adamsapple")

    sim_equal = scipy.spatial.distance.cosine(emb_apple, emb_apple)
    sim_apple = scipy.spatial.distance.cosine(emb_apple, emb_apples)
    sim_adams = scipy.spatial.distance.cosine(emb_apple, emb_adamsapple)

    print(1 - sim_equal)
    print(1 - sim_apple)
    print(1 - sim_adams)


if __name__ == "__main__":
    # test_embedding_generator()
    test_similarity()