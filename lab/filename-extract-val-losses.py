import pathlib
import re
from matplotlib import pyplot as plt


# Function to extract epoch and validation loss from file names
def extract_data(file_names):
    epochs = []
    losses = []
    for file_name in file_names:
        match = re.search(r"_e(\d+)_v([\d.]+)_", file_name)
        if match:
            epoch, loss = match.groups()
            epochs.append(int(epoch))
            losses.append(float(loss))
    return epochs, losses


def plot(results: dict):
    plt.figure(figsize=(10, 6))
    for name, result in results.items():
        epochs = result[0]
        losses = result[1]
        plt.plot(epochs, losses, label=name)

    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss Across Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    folders = {"csqa": "pth/to/csqa/results/folder",
               "merged": "pth/to/merged/results/folder"}

    results = {}
    for name, fldr in folders.items():
        files = pathlib.Path(fldr).glob("*.pth.tar")

        # Extracting data
        epochs, losses = extract_data(files)

        results[name] = (epochs, losses)

    print(results)
