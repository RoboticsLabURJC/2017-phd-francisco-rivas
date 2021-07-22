import os

import numpy as np
import yaml


class NetConfig:
    CLASSIFICATION_TYPE = "classification"
    REGRESSION_TYPE = "regression"

    def __init__(self, input_file):
        self.config_file = input_file
        if not os.path.exists(input_file):
            raise Exception("Input file {} does not exist".format(input_file))

        with open(input_file) as f:
            input_data = yaml.load(f, Loader=yaml.FullLoader)
            self.behaviour_name = list(input_data.keys())[0]
            print("LOADED CONFIG  ->    {}".format(self.behaviour_name))

            self._head_type = input_data[self.behaviour_name]["mode"]
            self._backend = input_data[self.behaviour_name]["backend"]
            self._fc_head = input_data[self.behaviour_name]["fc_head"]
            self._pretrained = input_data[self.behaviour_name]["pretrained"]
            self._weight_loss = input_data[self.behaviour_name]["weight_loss"]
            self._base_size = input_data[self.behaviour_name]["base_size"]
            self._batch_size = input_data[self.behaviour_name]["batch_size"]
            self._apply_vertical_flip = input_data[self.behaviour_name].get("apply_vertical_flip", True)
            self._non_common_samples_mult_factor = input_data[self.behaviour_name].get("non_common_samples_mult_factor", 0)
            self._loss_reduction = input_data[self.behaviour_name].get("loss_reduction", "mean")



            self._n_classes = None
            self._softmax_config = None

            if self._head_type == self.CLASSIFICATION_TYPE:
                classes_by_control = {}
                for control in ["w", "v"]:
                    if control in input_data[self.behaviour_name]["classification_data"]:
                        class_names = []
                        input_range = []
                        output_values = {}
                        for idx, slice in enumerate(input_data[self.behaviour_name]["classification_data"][control]):
                            slice_name = list(slice.keys())[0]
                            class_names.append(slice_name)
                            for value in slice[slice_name]["input"]:
                                if len(input_range) == 0 or value != input_range[-1]:
                                    input_range.append(value)
                            output_values[idx] = slice[slice_name]["output"]
                        classes_by_control[control] = {
                            "classes": class_names,
                            "input_config": input_range,
                            "output_config": output_values
                        }
                self.classification_data = classes_by_control
            else:
                self.regression_data = {"controllers": input_data[self.behaviour_name]["regression_data"]}
                self._n_classes = len(self.regression_data["controllers"])

    def encode(self, data):
        if self.head_type == self.CLASSIFICATION_TYPE:
            encoded_label = np.zeros(self.n_classes)  #
            for controller in self.classification_data:
                for idx in range(0, len(self.classification_data[controller]["input_config"]) - 1):
                    max_value = self.classification_data[controller]["input_config"][idx]
                    min_value = self.classification_data[controller]["input_config"][idx + 1]
                    if max_value == "None":
                        if data[controller] >= min_value:
                            encoded_label[self.get_global_idx(idx, controller)] = 1
                            break
                    elif min_value == "None":
                        if data[controller] <= max_value:
                            encoded_label[self.get_global_idx(idx, controller)] = 1
                            break
                    elif data[controller] < max_value and data[controller] >= min_value:
                        encoded_label[self.get_global_idx(idx, controller)] = 1
                        break

            return encoded_label
        else:
            output_data = []
            for controller in self.regression_data["controllers"]:
                output_data.append(data[controller])
            return np.asarray(output_data).astype(np.float32)

    def get_global_idx(self, idx, controller):
        if controller == "w":
            return idx
        else:
            if "w" not in self.classification_data:
                return idx
            else:
                return len(self.classification_data["w"]["classes"]) + idx

    def get_str_labels(self):
        output_labels = {}
        if self.head_type == self.CLASSIFICATION_TYPE:
            for controller in self.classification_data:
                output_labels[controller] = self.classification_data[controller]["classes"]
        return output_labels

    def get_real_values_from_estimation(self, estimation):
        output_data = {}
        if self.head_type == self.CLASSIFICATION_TYPE:
            for controller in self.classification_data:
                output_data[controller] = self.classification_data[controller]["output_config"][estimation[controller]]
        return output_data

    @property
    def n_classes(self):
        if self._n_classes is None:
            if self.head_type == self.CLASSIFICATION_TYPE:
                total_classes = 0
                for controller in self.classification_data:
                    total_classes += len(self.classification_data[controller]["classes"])
                self._n_classes = total_classes

        return self._n_classes

    @property
    def softmax_config(self):
        if self._softmax_config is None:
            idx_base = 0
            self._softmax_config = {}
            for controller in self.classification_data:
                self._softmax_config[controller] = list(
                    range(0 + idx_base, len(self.classification_data[controller]["classes"]) + idx_base))
                idx_base += len(self.classification_data[controller]["classes"])
        return self._softmax_config

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def head_type(self):
        return self._head_type

    @property
    def fc_head(self):
        return self._fc_head

    @property
    def backend(self):
        return self._backend

    @property
    def pretrained(self):
        return self._pretrained

    @property
    def weight_loss(self):
        return self._weight_loss

    @property
    def base_size(self):
        return self._base_size

    @property
    def apply_vertical_flip(self):
        return self._apply_vertical_flip

    @property
    def non_common_samples_mult_factor(self):
        return self._non_common_samples_mult_factor

    @property
    def loss_reduction(self):
        return self._loss_reduction