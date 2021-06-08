from visual_control_utils.fix_json import read_malformed_json
import os


def load_dataset(dataset_path, path_name, data_json_file):
    train_data = read_malformed_json(os.path.join(dataset_path, path_name, data_json_file))
    train_images_path = os.path.join(dataset_path, path_name, "Images")
    image_file = os.listdir(train_images_path)
    train_images = sorted(image_file, key=lambda x: int(x.split(".")[0]))

    return train_data, train_images
