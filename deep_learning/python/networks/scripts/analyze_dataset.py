from visual_control_utils.visual_datset_format import load_dataset
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == "__main__":
    dataset_path = "/home/frivas/Descargas/complete_dataset"

    train_data, train_images = load_dataset(dataset_path, "Train", "train.json")

    df = pd.DataFrame()
    for idx, label in enumerate(train_data):
        df.loc[idx, "v"] = label["v"]
        df.loc[idx, "w"] = label["w"]

    print(df.describe())
    ax = sns.displot(df['v'])
    ax2 = sns.displot(df['w'])

    plt.tight_layout()
    plt.show()
