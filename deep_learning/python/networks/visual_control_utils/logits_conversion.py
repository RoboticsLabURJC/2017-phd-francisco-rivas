import torch

from dataloaders.visual_control_encoder import VisualControlEncoder

import numpy as np


def from_logit_to_estimation(logit, encoder:VisualControlEncoder):
    output = np.array([])
    for output_key in ["w", "v"]:
        if output_key in encoder.softmax_config:
            logit_position = encoder.softmax_config[output_key]
            estimation = torch.softmax(torch.FloatTensor([logit[i] for i in logit_position]), 0)
            one_hot_estimation = (estimation.numpy() == np.max(estimation.numpy())).astype(np.float)
            output = np.concatenate((output, one_hot_estimation), axis=0)
    return output


def from_logit_to_estimation_class(logit, encoder:VisualControlEncoder):
    output = {}
    for output_key in ["w", "v"]:
        if output_key in encoder.softmax_config:
            logit_position = encoder.softmax_config[output_key]
            estimation = torch.softmax(torch.FloatTensor([logit[i] for i in logit_position]), 0)
            class_idx_estimation = np.argmax(estimation.numpy(), axis=0)
            output[output_key] = class_idx_estimation
    return output



def from_one_hot_to_class(encoded, encoder:VisualControlEncoder):
    output = {}
    for output_key in ["w", "v"]:
        if output_key in encoder.softmax_config:
            logit_position = encoder.softmax_config[output_key]
            class_idx_estimation = np.argmax([encoded[i] for i in logit_position], axis=0)
            output[output_key] = class_idx_estimation
    return output