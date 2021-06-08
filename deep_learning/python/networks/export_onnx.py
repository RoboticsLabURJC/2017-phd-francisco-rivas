import json
import os
import cv2
import numpy as np
import albumentations as A
import torchvision
from albumentations.pytorch import ToTensorV2
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm
import pandas as pd
from utils.plot_utils import plot_confusion_matrix
from sklearn.metrics import accuracy_score
import torch

from sklearn.metrics import confusion_matrix

from dataloaders.visual_control_encoder import VisualControlEncoder
from models.visual_control import VisualControl
from utils.logits_conversion import from_logit_to_estimation, from_one_hot_to_class

from utils.visual_datset_format import load_dataset
from utils.visualization import add_labels_to_image
import matplotlib.pyplot as plt

if __name__ == "__main__":
    dataset_path = "/home/frivas/Descargas/complete_dataset/"
    checkpoint_path = "/home/frivas/devel/mio/github/2017-phd-francisco-rivas/deep_learning/python/networks/lightning_logs/version_11/checkpoints/rc-classification-epoch=04-val_acc=0.98-val_loss=0.06.ckpt"
    checkpoint_path = "/home/frivas/devel/mio/github/2017-phd-francisco-rivas/deep_learning/python/networks/lightning_logs/version_16/checkpoints/rc-classification-epoch=00-val_acc=0.88-val_loss=0.32.ckpt"

    test_data, test_images = load_dataset(dataset_path, "Test", "test.json")

    test_dict = {
        "labels": test_data,
        "images": test_images
    }

    base_size = 418
    batch_size = 50


    model = VisualControl.load_from_checkpoint(checkpoint_path=checkpoint_path, dataset_path=dataset_path, lr=5e-2,
                          base_size=base_size, dataset_encoder_type=VisualControlEncoder.WV_3_2_CLASSES, batch_size=batch_size)

    encoder = VisualControlEncoder(VisualControlEncoder.WV_3_2_CLASSES)
    # prints the learning_rate you used in this checkpoint

    eval_transforms = A.Compose([
        A.LongestMaxSize(base_size, always_apply=True),
        A.PadIfNeeded(base_size, base_size, always_apply=True,
                      border_mode=cv2.BORDER_CONSTANT),
        A.Normalize(),
        ToTensorV2()
    ])

    device = "cuda:0"

    model.to(device)
    model.eval()
    model.freeze()

    results = {}

    image_path = os.path.join(dataset_path, "Test", "Images", test_dict["images"][0])
    image = cv2.imread(image_path)
    x = eval_transforms(image=image)
    input = x['image'].unsqueeze(0).to(device)

    model = torchvision.models.mobilenet_v2(pretrained=True)
    input = torch.randn(2, 3, 224, 224, requires_grad=True)
    torch.onnx.export(model, (input,), './large_model.onnx', use_external_data_format=True)

    # torch.onnx.export(model,  # model being run
    #                   input,  # model input (or a tuple for multiple inputs)
    #                   "visual_control_fran.onnx",  # where to save the model (can be a file or file-like object)
    #                   export_params=True,  # store the trained parameter weights inside the model file
    #                   opset_version=11,  # the ONNX version to export the model to
    #                   do_constant_folding=True,  # whether to execute constant folding for optimization
    #                   input_names=['input'],  # the model's input names
    #                   output_names=['output'],  # the model's output names
    #                   dynamic_axes={'input': {2: '?', 3: '?'},  # variable length axes
    #                                 'output': {0: '?'},
    #                                 }
    #                   )