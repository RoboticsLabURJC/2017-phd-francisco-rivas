import cv2
import numpy as np
import math

def add_labels_to_image(image:np.ndarray, gt: list, pred: list):
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    font_scale = 0.5

    cv2.putText(image, "-GT - {}".format(gt), (10, 20), font, font_scale, (255, 255, 0), thickness)
    cv2.putText(image, "-P  - {}".format(pred), (10, 40), font, font_scale, (255, 0, 255), thickness)
    return image


def add_arrow_prediction(image:np.ndarray, pred:dict):
    box_size = 100
    arrow_box = [0, image.shape[0] - box_size, box_size*2, image.shape[0]]

    cv2.rectangle(image, arrow_box[0:2], arrow_box[2:], (255,0,0))

    signed = 1 if pred["w"] < 0 else -1
    angle = 90 - ((abs(pred["w"])/2.2) * 90)

    h = np.arctan(math.radians(angle))

    arrow_origin = np.array((box_size, image.shape[0]))

    if angle == 90:
        new_arrow_destination = np.array((box_size, image.shape[0] - box_size))
    else:
        if signed == 1:
            arrow_destination = np.array([box_size + 1, image.shape[0] - h])
        else:
            arrow_destination = np.array([box_size - 1, image.shape[0] - h])

        arrow = arrow_destination - arrow_origin
        u_arrow = arrow / np.linalg.norm(arrow)
        new_arrow_destination = u_arrow * box_size
        new_arrow_destination = arrow_origin + new_arrow_destination

    cv2.arrowedLine(image, arrow_origin.tolist(), new_arrow_destination.astype(np.int32).tolist(), (255,255,0))

    return image