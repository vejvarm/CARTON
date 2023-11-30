# CARTON w/ Named Entity Recognition
- inspired by and forked from [endrikacupaj/CARTON](https://github.com/endrikacupaj/CARTON.git)
- inspired by [endrikacupaj/LASAGNE](https://github.com/endrikacupaj/LASAGNE)

## Architecture
![CARTONNER](image/CARTONNER.png?raw=true "CARTONNER architecture")

## Premise
Expand CARTON beyond static knowledge and allow it to work with dynamic Knolwedge Graphs.
Retrain with new dataset and new `Insert` class (logical form action type), which will update the underlying Knowledge Graph with new RDF triples based on information from the user input sentence.


Below is a work in progress Readme adapted from the original repository.

# Context Transformer with Stacked Pointer Networks for Conversational Question Answering over Knowledge Graphs

*Re-implementation using code base and grammar from [here](https://github.com/endrikacupaj/LASAGNE). Original implementation was made by [joanPlepi](https://github.com/joanPlepi).*

Neural semantic parsing approaches have been widely used for Question Answering (QA) systems over knowledge graphs. Such methods provide the flexibility to handle QA datasets with complex queries and a large number of entities. In this work, we propose a novel framework named CARTON, which performs multi-task semantic parsing for handling the problem of conversational question answering over a large-scale knowledge graph. Our framework consists of a stack of pointer networks as an extension of a context transformer model for parsing the input question and the dialog history. The framework generates a sequence of actions that can be executed on the knowledge graph. We evaluate CARTON on a standard dataset for complex sequential question answering on which CARTON outperforms all baselines. Specifically, we observe performance improvements in F1-score on eight out of ten question types compared to the previous state of the art. For logical reasoning questions, an improvement of 11 absolute points is reached.

![CARTON](image/carton_architecture.png?raw=true "CARTON architecture")

CARTON (Context Transformer with Stacked Pointer Networks architecture) architecture. It consists of three modules: 1) A Transformer-based contextual encoder which produces the representation of the current context of the dialogue. 2) A logical form decoder that generates the pattern of actions defined by our proposed grammar. 3) The stacked pointer networks that initialize the KG items to fetch the correct answer.

## Requirements and Setup
Python version >= 3.9
PyTorch version >= 1.12.0

``` bash
# clone the repository
git clone https://github.com/vejvarm/CARTONNER.git
cd CARTONNER
pip install -r requirements.txt
```
## Dataset
### CSQA
The original framework was evaluated on [CSQA](https://amritasaha1812.github.io/CSQA/) dataset. You can download the dataset from [here](https://amritasaha1812.github.io/CSQA/download/).

### CSQA-D2T
We expand the CSQA dataset with a new `Insert` action class entries. CSQA-D2T:
- uses artificially generated declarative sentences from factual data from the Wikidata KG
- aims to train `Insert` action, which will generate new RDF triple in the underlying KG
- follows same sturcture as CSQA dataset. 
- is generated using
  - [(2022, Kasner & Dušek, Neural Pipeline)](https://github.com/kasnerz/zeroshot-d2t-pipeline)
  - [(2023, Vejvar & Fujimoto, ASPIRO)](https://github.com/vejvarm/ASPIRO)
## Wikidata Knowlegde Graph
Since CSQA is based on Wikidata [Knowlegde Graph](https://www.wikidata.org/wiki/Wikidata:Main_Page), the authors provide a preproccesed version of it which can be used when working with the dataset.
You can download the preprocessed files from [here](https://zenodo.org/record/4052427#.YBU7xHdKjfZ).
After dowloading you will need to move them under the [knowledge_graph](knowledge_graph) directory.

## Prepare Wikidata Knowlegde Graph Files
We prefer to merge some JSON files from the preprocessed Wikidata, for accelerating the process of reading all the knowledge graph files. In particular, we create three new JSON files using the script [prepare_data.py](scripts/prepare_data.py). Please execute the script as below.
``` bash
# prepare knowlegde graph files
python scripts/prepare_data.py
```

## Annotate Dataset
Next, using the preproccesed Wikidata files we can annotate CSQA dataset with our proposed grammar.
``` bash
# annotate CSQA dataset with proposed grammar
python annotate_csqa/preprocess.py --partition train --annotation_task actions --read_folder /path/to/CSQA --write_folder /path/to/write
```

## Create BERT embeddings
Before training the framework, we need to create BERT embeddings for all the knowledge graph entities. You can do that by running.
``` bash
# create bert embeddings
python scripts/bert_embeddings.py
```

## Train Framework
For training you will need to adjust the paths in [args](args.py) file. At the same file you can also modify and experiment with different model settings.
``` bash
# train framework
python train.py
```

## Inference Framework 
Calculates accuracy and recall on test split
- accuracy averaging: 'micro'
- recal averaging: 'macro'
``` bash
python inference.py --name "00_csqa_on_merged" --batch-size 40 --model-path experiments/models/CARTONNER_csqa_e10_v0.0102_multitask.pth.tar --data-path data/csqa-merged --cache-path .cache/merged/
```
will save metric results as JSON files into `ROOT_PATH/args.path_inference/args.name` folder.

## Generate Actions
After the model has finished training we perform the inference in 2 steps.
First, we generate the actions and save them in JSON file using the trained model.
``` bash
# generate actions for a specific question type
python test.py --question_type Clarification
```

## Execute Actions
Second, we execute the actions and get the results from Wikidata files.
``` bash
# execute actions for a specific question type
python action_executor/run.py --file_path /path/to/actions.json --question_type Clarification
```

## Changes
### Generating BERT embeddings on the fly for new labels
New file `embeddings.py` and new class `EmbeddingGenerator` to manage embeddings generated by BERT.
``` python
eg = EmbeddingGenerator([path_to_embedding_database_file])
id, label, emb = eg.add_entry(label)  # -> (id: str, label: str, emb: np.ndarray)
```
__if label exists in `database_file`__: return the existing (`id`, `label`, `emb`) tuple \
__else__: generate new `id` and `emb`, add it to `database_file` and return (`id`, `label`, `emb`)

## License
The repository is under [MIT License](LICENCE).

## Cite 
### CARTON
```bash
@InProceedings{10.1007/978-3-030-77385-4_21,
author="Plepi, Joan
and Kacupaj, Endri
and Singh, Kuldeep
and Thakkar, Harsh
and Lehmann, Jens",
editor="Verborgh, Ruben
and Hose, Katja
and Paulheim, Heiko
and Champin, Pierre-Antoine
and Maleshkova, Maria
and Corcho, Oscar
and Ristoski, Petar
and Alam, Mehwish",
title="Context Transformer with Stacked Pointer Networks for Conversational Question Answering over Knowledge Graphs",
booktitle="The Semantic Web",
year="2021",
publisher="Springer International Publishing",
address="Cham",
pages="356--371",
abstract="Neural semantic parsing approaches have been widely used for Question Answering (QA) systems over knowledge graphs. Such methods provide the flexibility to handle QA datasets with complex queries and a large number of entities. In this work, we propose a novel framework named CARTON (Context trAnsformeR sTacked pOinter Networks), which performs multi-task semantic parsing for handling the problem of conversational question answering over a large-scale knowledge graph. Our framework consists of a stack of pointer networks as an extension of a context transformer model for parsing the input question and the dialog history. The framework generates a sequence of actions that can be executed on the knowledge graph. We evaluate CARTON on a standard dataset for complex sequential question answering on which CARTON outperforms all baselines. Specifically, we observe performance improvements in F1-score on eight out of ten question types compared to the previous state of the art. For logical reasoning questions, an improvement of 11 absolute points is reached.",
isbn="978-3-030-77385-4"
}
```
