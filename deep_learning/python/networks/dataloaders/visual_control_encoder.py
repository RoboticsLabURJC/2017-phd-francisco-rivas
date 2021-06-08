import numpy as np

class VisualControlEncoder:
    WV_3_2_CLASSES = "WV_3_2_CLASSES"  # V
    WV_9_5_CLASSES = "WV_9_5_CLASSES"  # V
    WV_5_3_CLASSES = "WV_5_3_CLASSES"  # V
    WV_1_1_CLASSES = "WV_1_1_CLASSES"

    CLASSIFICATION_TYPE = "CLASSIFICATION_TYPE"
    REGRESSION_TYPE = "REGRESSION_TYPE"


    VALID_ENCODERS = [ WV_3_2_CLASSES, WV_9_5_CLASSES, WV_5_3_CLASSES, WV_1_1_CLASSES]

    def __init__(self, encoder):
        if encoder not in self.VALID_ENCODERS:
            raise Exception("Encoding type {} not supported ({})".format(encoder, self.VALID_ENCODERS))

        self.encoder = encoder
        self.n_classes = None
        self.softmax_config = None
        self.head_type = None

        if self.encoder == self.WV_3_2_CLASSES:
            self.n_classes = 5
            self.softmax_config = {
                "w": [0, 1, 2],
                "v": [3, 4]
            }
            self.head_type = self.CLASSIFICATION_TYPE
        elif self.encoder == self.WV_9_5_CLASSES:
            self.n_classes = 14
            self.softmax_config = {
                "w": [0, 1, 2, 3, 4, 5, 6, 7, 8],
                "v": [9, 10, 11, 12, 13]
            }
            self.head_type = self.CLASSIFICATION_TYPE
        elif self.encoder == self.WV_5_3_CLASSES:
            self.n_classes = 8
            self.softmax_config = {
                "w": [0, 1, 2, 3, 4],
                "v": [5,6,7]
            }
            self.head_type = self.CLASSIFICATION_TYPE
        elif self.encoder == self.WV_1_1_CLASSES:
            self.n_classes = 2
            self.head_type = self.REGRESSION_TYPE



    def encode(self, data):
        encoded_label = np.zeros(self.n_classes)  #
        if self.encoder == self.WV_3_2_CLASSES:
            if data["w"] > 0.2:
                encoded_label[0] = 1
            elif 0.2 >= data["w"] >= -0.2:
                encoded_label[1] = 1
            elif data["w"] < -0.2:
                encoded_label[2] = 1
            if data["v"] < 9:
                encoded_label[3] = 1
            elif data["v"] >= 9:
                encoded_label[4] = 1
        elif self.encoder == self.WV_9_5_CLASSES:
            if data["w"] >= 2:
                encoded_label[0] = 1
            elif 2 > data["w"] >= 1:
                encoded_label[1] = 1
            elif 1 > data["w"] >= 0.5:
                encoded_label[2] = 1
            elif 0.5 > data["w"] >= 0.1:
                encoded_label[3] = 1
            elif 0.1 > data["w"] >= -0.1:
                encoded_label[4] = 1
            elif -0.1 > data["w"] >= -0.5:
                encoded_label[5] = 1
            elif -0.5 > data["w"] >= -1:
                encoded_label[6] = 1
            elif -1 > data["w"] >= -2:
                encoded_label[7] = 1
            elif data["w"] < -2:
                encoded_label[8] = 1

            if data["v"] < 0:
                encoded_label[9] = 1
            elif 0 <= data["v"] < 5:
                encoded_label[10] = 1
            elif 5 <= data["v"] < 9:
                encoded_label[11] = 1
            elif 9 <= data["v"] < 11:
                encoded_label[12] = 1
            elif data["v"] >= 11:
                encoded_label[13] = 1
        elif self.encoder == self.WV_5_3_CLASSES:
            if data["w"] >= 0.5:
                encoded_label[0] = 1
            elif 0.5 > data["w"] >= 0.2:
                encoded_label[1] = 1
            elif 0.2 > data["w"] >= -0.2:
                encoded_label[2] = 1
            elif -0.2 > data["w"] >= -0.5:
                encoded_label[3] = 1
            elif data["w"] < -0.5:
                encoded_label[4] = 1

            if data["v"] <= 5:
                encoded_label[5] = 1
            elif 5 < data["v"] < 10:
                encoded_label[6] = 1
            elif data["v"] >= 10:
                encoded_label[7] = 1
        elif self.encoder == self.WV_1_1_CLASSES:
            encoded_label[0] = data["w"]
            encoded_label[1] = data["v"]

            encoded_label = encoded_label.astype(np.float32)


        return encoded_label

    def get_str_labels(self):
        if self.encoder == self.WV_3_2_CLASSES:
            return {
                "w": ["left", "slight", "right"],
                "v": ["moderate", "fast"]
            }
        elif self.encoder == self.WV_9_5_CLASSES:
            return {
                "w": ["radically_left", "strongly_left", "moderately_left", "slightly_left", "slight", "slightly_right", "moderately_right", "strongly_right", "radically_right"],
                "v": ["negative", "slow", "moderate", "fast", "very_fast"]
            }

    def get_real_values_from_estimation(self, estimation):
        v = None
        w = None
        if self.encoder == self.WV_3_2_CLASSES:
            if estimation["w"] == 0:
                w = 1
            elif estimation["w"] == 1:
                w = 0
            elif estimation["w"] == 2:
                w = -1

            if estimation["v"] == 0:
                v = 5
            elif estimation["v"] == 1:
                v = 10
        elif self.encoder == self.WV_9_5_CLASSES:
            if estimation["w"] == 0:
                w = 1
            elif estimation["w"] == 1:
                w = 1
            elif estimation["w"] == 2:
                w = 0
            elif estimation["w"] == 3:
                w = 0
            elif estimation["w"] == 4:
                w = 0
            elif estimation["w"] == 5:
                w = 0
            elif estimation["w"] == 6:
                w = 0
            elif estimation["w"] == 7:
                w = -1
            elif estimation["w"] == 8:
                w = -1

            if estimation["v"] == 0:
                v = 5
            elif estimation["v"] == 1:
                v = 5
            elif estimation["v"] == 2:
                v = 5
            elif estimation["v"] == 3:
                v = 10
            elif estimation["v"] == 4:
                v = 10
        elif self.encoder == self.WV_5_3_CLASSES:
            if estimation["w"] == 0:
                w = 1
            elif estimation["w"] == 1:
                w = 1
            elif estimation["w"] == 2:
                w = 0
            elif estimation["w"] == 3:
                w = -1
            elif estimation["w"] == 4:
                w = -1
            if estimation["v"] == 0:
                v = 2
            elif estimation["v"] == 1:
                v = 2
            elif estimation["v"] == 2:
                v = 2


        return {"w": w, "v": v}
    @staticmethod
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def get_smooth_estimation(self, estimation):
        w = 0
        v = 0
        if self.encoder == self.WV_9_5_CLASSES:
            w_values = [1, 1, 1, 0, 0, 0, -1, -1, -1]
            v_values = [5, 5, 5, 10, 10]
            w = np.sum(self.softmax(estimation[0:9]) * np.array(w_values))
            v = np.sum(self.softmax(estimation[9:]) * np.array(v_values))

        return {"w": w, "v": v}

