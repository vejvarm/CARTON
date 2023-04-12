# RDF Dataset Generation for D2T Generation

This script generates a dataset of RDF entries, aggregated by having the same subject entity, for the purpose of D2T generation from the extracted RDF files.

## Purpose

- Connect to an Elasticsearch index containing RDF-style entries with subject (sid), relation (rid), and object (oid) triplets.
- Aggregate and process the entries based on their subject entity.
- Split the data into train, test, and dev sets.
- Save the processed data into a specific folder structure.

## Methods

- Connect to Elasticsearch and fetch the required data using composite aggregations.
- Use a helper function to process and save the triples for train, test, and dev sets.

## Output Format

- Each set of triples is saved in a JSON file with the following structure:


``` json
{
  "data": [
    [
      "subject1_label | relation_label | object_label",
      "subject1_label | ... | ...",
      "subject1_label | ... | ..."
    ]
  ]
}
```

## File System Folder Structure
The generated files are saved in the following folder structure:

    text_generation/data/buckets: Contains JSON files with unprocessed buckets.
    text_generation/data/train: Contains subdirectories for each triplet count (e.g., 1triples, 2triples, etc.), which store JSON files for the train set.
    text_generation/data/test: Contains subdirectories for each triplet count (e.g., 1triples, 2triples, etc.), which store JSON files for the test set.
    text_generation/data/dev: Contains subdirectories for each triplet count (e.g., 1triples, 2triples, etc.), which store JSON files for the dev set.