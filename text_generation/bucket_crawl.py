from pathlib import Path
import json

path_to_bucket_file = Path("./buckets.json")

if __name__ == "__main__":
    buckets = json.load(path_to_bucket_file.open("r"))
    i = 1
    print(buckets[i])
    print(len(buckets[i]["hits"]["hits"]["hits"]))