from dataset import CSQADataset
from args import get_parser

parser = get_parser()
args = parser.parse_args()


def main():
    # load data
    dataset = CSQADataset(args)  # load all data from all splits to build full vocab from all splits
    return dataset, dataset.build_vocabs(args.stream_data)
    # data_dict, helper_dict = dataset.preprocess_data()


if __name__ == "__main__":
    dataset, vocabs = main()
