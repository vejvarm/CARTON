from functools import partial
import torch
import pandas as pd
from torch import cosine_similarity
from torchtext.vocab import GloVe

from constants import ROOT_PATH

if __name__ == '__main__':
    glove_vectors = GloVe(cache=f'{ROOT_PATH}/.vector_cache')

    sentences = [
        "The sky is blue",
        "The sun is bright today",
        "The sun in the sky is bright",
        "We can see the shining sun, the bright sun",
    ]

    D = [glove_vectors.get_vecs_by_tokens(s.split()) for s in sentences]
    D_bag = list(map(partial(torch.mean, dim=0), D))

    # D_bag = [torch.Tensor([0.301, 0, 0, 0, 0, 0.151, 0, 0]),
    #          torch.Tensor([0, 0.0417, 0, 0, 0, 0, 0.0417, 0.201]),
    #          torch.Tensor([0, 0.0417, 0, 0, 0, 0.1, 0.0417, 0]),
    #          torch.Tensor([0, 0.0209, 0.1, 0.1, 0.1, 0, 0.0417, 0])]

    rows = []
    for i, da in enumerate(D_bag):
        rows.append([d.cpu().numpy() for d in map(partial(cosine_similarity, da, dim=0), D_bag)])

    index = [f'd{i+1}' for i in range(len(D))]
    df = pd.DataFrame(rows, index=index, columns=index)

    df.to_csv(f'{ROOT_PATH}/experiments/results/glove_emb_cos_similarity.csv')