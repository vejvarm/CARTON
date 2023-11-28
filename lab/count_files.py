import pathlib
import argparse


def count_files(root: pathlib.Path, glob_pattern="**/QA_*.json"):
    paths = root.glob(glob_pattern)

    count = 0
    for _ in paths:
        count += 1

    return count


def main(args):
    folder0 = pathlib.Path(args.folder0)
    folder1 = pathlib.Path(args.folder1)
    folder2 = pathlib.Path(args.folder2)

    count0 = count_files(folder0)
    count1 = count_files(folder1)
    count2 = count_files(folder2)

    counts = {folder0.name: count0,
              folder1.name: count1,
              folder2.name: count2}

    print(counts)
    print(
        f"{folder0.name}: {count0} | {folder1.name}+{folder2.name}: {count1 + count2}\ndiff: {count0 - count1 - count2}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Count number of QA_*.json files in folder0, folder1, folder2 and compare folder0 ~ folder1+folder2")

    parser.add_argument("--folder0", required=True, type=str)
    parser.add_argument("--folder1", required=True, type=str)
    parser.add_argument("--folder2", required=True, type=str)

    main(parser.parse_args())
