import numpy as np
from matplotlib import pyplot as plt

from constants import ROOT_PATH
from args import get_parser

# read parser
parser = get_parser()
args = parser.parse_args()

def extract_val_loss_from_train_log(file_name):
    with open(f"{ROOT_PATH}/{args.path_results}/{file_name}", 'r') as f:
        out = []
        epoch = 0
        for line in f:
            if "Val loss:" in line:
                line_split = line.split("Val loss: ")
                out.append(float(line_split[-1]))
                epoch += 1

        arr = np.array(out)

    with open(f"{args.path_results}/out_{file_name.split('.')[0]}.npy", 'wb') as f:
        np.save(f, arr)


def plot_from_np_file(path_to_np_file, out_plot_file_path):
    arr = np.load(path_to_np_file)
    plt.plot(arr)
    plt.title("Training validation loss")
    plt.xlabel("epoch")
    plt.ylabel("validation loss")
    plt.savefig(out_plot_file_path, format="png", dpi=200)


if __name__ == '__main__':
    # log_file_name = "train_multitask.log"
    # extract_val_loss_from_train_log(log_file_name)

    input_np_file_path = f"{args.path_results}/out_train_multitask.npy"
    output_plot_file_path = f"{args.path_results}/val_loss.png"
    plot_from_np_file(input_np_file_path, output_plot_file_path)

