from net_config.net_config import NetConfig
from visual_control_utils.fix_json import read_malformed_json
import os
import pandas as pd

def load_dataset(dataset_path, path_name, data_json_file):
    train_data = read_malformed_json(os.path.join(dataset_path, path_name, data_json_file))
    train_images_path = os.path.join(dataset_path, path_name, "Images")
    image_file = os.listdir(train_images_path)
    train_images = sorted(image_file, key=lambda x: int(x.split(".")[0]))

    return train_data, train_images



def load_dataset_new(dataset_path, net_config:NetConfig, split:str="train"):

    if split == "train":
        data_to_load = [
            "difficult_situations_01_04_2022",
            "difficult_situations_01_04_2022_2",
            "extended_simple_circuit_01_04_2022_clockwise_1",
            "many_curves_01_04_2022_clockwise_1",
            "monaco_01_04_2022_clockwise_1",
            "nurburgring_01_04_2022_clockwise_1",
            "only_curves_01_04_2022"
        ]
    else:
        data_to_load = [
            "simple_circuit_01_04_2022_anticlockwise_1",
            "simple_circuit_01_04_2022_clockwise_1",
            "montreal_12_05_2022_opencv_anticlockwise_1",
            "montreal_12_05_2022_opencv_clockwise_1",
            "montmelo_12_05_2022_opencv_anticlockwise_1",
            "montmelo_12_05_2022_opencv_clockwise_1"
        ]
    train_data = []
    train_images = []

    w_mean, w_std, v_mean, v_std = net_config.norm_values

    for file_data in data_to_load:
        if "difficult" in file_data or "only_curves" in file_data:
            all_files_to_process = os.listdir(os.path.join(dataset_path, file_data))
            current_dataset_path = os.path.join(dataset_path, file_data)
        else:
            all_files_to_process = [file_data]
            current_dataset_path = dataset_path
        for single_path in all_files_to_process:
            if "checkpoint" in single_path:
                continue
            df = pd.read_csv(os.path.join(current_dataset_path, single_path, "data.csv"))
            for idx, row in df.iterrows():
                w = (float(row["w"]) - w_mean) / w_std
                v = (float(row["v"]) - v_mean) / v_std

                train_data.append(
                    {"w": w,
                     "v": v
                     }
                )
                train_images.append(os.path.join(current_dataset_path, single_path, row["image_name"]))

    return train_data, train_images