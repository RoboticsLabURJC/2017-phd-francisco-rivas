import json
import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm
import pandas as pd
from visual_control_utils.plot_utils import plot_confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from dataloaders.visual_control_encoder import VisualControlEncoder
from models.visual_control import VisualControl
from visual_control_utils.logits_conversion import from_logit_to_estimation, from_one_hot_to_class

from visual_control_utils.visual_datset_format import load_dataset
from visual_control_utils.visualization import add_labels_to_image, add_arrow_prediction
import matplotlib.pyplot as plt

if __name__ == "__main__":
    dataset_path = "/home/frivas/Descargas/complete_dataset/"
    checkpoint_path = "/home/frivas/devel/mio/github/2017-phd-francisco-rivas/deep_learning/python/networks/lightning_logs/version_11/checkpoints/rc-classification-epoch=04-val_acc=0.98-val_loss=0.06.ckpt"
    checkpoint_path = "/home/frivas/devel/mio/github/2017-phd-francisco-rivas/deep_learning/python/networks/lightning_logs/version_20/checkpoints/rc-classification-epoch=32-val_acc=0.95-val_loss=0.05-v1.ckpt"
    checkpoint_path = "/home/frivas/devel/mio/github/2017-phd-francisco-rivas/deep_learning/python/networks/lightning_logs/version_24/checkpoints/rc-classification-epoch=04-val_acc=0.92-val_loss=0.08.ckpt"
    checkpoint_path = "/home/frivas/devel/mio/github/2017-phd-francisco-rivas/deep_learning/python/networks/lightning_logs/version_25/checkpoints/rc-classification-epoch=51-val_acc=0.97-val_loss=0.05.ckpt"
    checkpoint_path = "/home/frivas/devel/mio/github/2017-phd-francisco-rivas/deep_learning/python/networks/lightning_logs/version_27/checkpoints/rc-classification-epoch=104-val_acc=0.98-val_loss=0.09.ckpt"
    checkpoint_path = "/home/frivas/devel/mio/github/2017-phd-francisco-rivas/deep_learning/python/networks/lightning_logs/version_33/checkpoints/rc-classification-epoch=95-rmse=0.44-val_loss=0.29.ckpt"

    test_data, test_images = load_dataset(dataset_path, "Test", "test.json")

    test_dict = {
        "labels": test_data,
        "images": test_images
    }

    base_size = 418
    batch_size = 1

    encoder_type = VisualControlEncoder.WV_1_1_CLASSES

    model = VisualControl.load_from_checkpoint(checkpoint_path=checkpoint_path, dataset_path=dataset_path, lr=5e-2,
                          base_size=base_size, dataset_encoder_type=encoder_type, batch_size=batch_size)

    encoder = VisualControlEncoder(encoder_type)
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



    visualize = True
    save = False
    output_dir = "visual_results"
    if save:
        if os.path.exists(output_dir):
            raise Exception("output dir already exists")
        os.mkdir(output_dir)

    for batch_idx in tqdm(range(0, len(test_dict["labels"]), batch_size)):
        images_batch = []
        labels = []
        raw_images = []
        for idx in range(batch_idx, min(batch_idx + batch_size, len(test_dict["labels"]))):
            image_path = os.path.join(dataset_path, "Test", "Images", test_dict["images"][idx])
            image = cv2.imread(image_path)
            raw_images.append(image)
            x = eval_transforms(image=image)
            y = encoder.encode(test_dict["labels"][idx])
            images_batch.append(x["image"])
            labels.append(y)


        predictions = model(default_collate(images_batch).to(device))
        for idx in range(0, len(predictions)):

            # d = encoder.get_smooth_estimation(predictions[idx].cpu().numpy())

            if encoder.head_type == VisualControlEncoder.CLASSIFICATION_TYPE:
                y_hat = from_logit_to_estimation(predictions[idx], encoder)
            else:
                y_hat = predictions[idx].cpu().numpy()
            y = labels[idx]
            results[idx+batch_idx] = {
                "label": y,
                "prediction": y_hat
            }

            if visualize or save:
                if visualize:
                    print("-----------------")
                    print(y)
                    print(y_hat)
                image_labels = add_labels_to_image(raw_images[idx], y, y_hat)

                if encoder.head_type == VisualControlEncoder.CLASSIFICATION_TYPE:
                    class_idx = from_one_hot_to_class(y_hat, encoder)
                    motors_info = encoder.get_real_values_from_estimation(class_idx)
                else:
                    motors_info = {"w": y_hat[0], "v": y_hat[1]}
                image_labels = add_arrow_prediction(image_labels, motors_info)

                if visualize:
                    cv2.imshow("Test", image_labels)
                    cv2.waitKey(0)
                if save:
                    output_file = os.path.join(output_dir, "{}.jpg".format(idx+batch_idx))
                    cv2.imwrite(output_file, image_labels)

    gt = [ results[x]["label"] for x in results ]
    pred = [ results[x]["prediction"] for x in results ]
    acc = accuracy_score(gt, pred)
    acc2 = accuracy_score(gt, pred, normalize=False)
    print(acc)
    print("{}/{}".format(acc2, len(gt)))


    #confusion matrices

    final_data = {

    }
    for current_data in results:
        gt = results[current_data]["label"]
        pred = results[current_data]["prediction"]
        class_data_gt = from_one_hot_to_class(gt, encoder)
        class_data_pred = from_one_hot_to_class(pred, encoder)

        for output_key in class_data_gt:
            if output_key not in final_data:
                final_data[output_key] = {
                    "label": [],
                    "prediction": []
                }
            final_data[output_key]["label"].append(class_data_gt[output_key])
            final_data[output_key]["prediction"].append(class_data_pred[output_key])

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(15, 15, forward=True)

    for idx, output_key in enumerate(final_data):
        ax = axs[idx]
        a = confusion_matrix(final_data[output_key]["label"], final_data[output_key]["prediction"], normalize="true")
        plot_confusion_matrix(a, classes=encoder.get_str_labels()[output_key], title=output_key, ax=ax)

    plt.tight_layout()
    plt.show()
