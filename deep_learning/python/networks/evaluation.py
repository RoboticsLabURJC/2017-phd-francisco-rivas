import json
import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm
import pandas as pd

from visual_control_utils.check_point_loader import load_best_model
from visual_control_utils.plot_utils import plot_confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from net_config.net_config import NetConfig
from models.visual_control import VisualControl
from visual_control_utils.logits_conversion import from_logit_to_estimation, from_one_hot_to_class

from visual_control_utils.visual_datset_format import load_dataset, load_dataset_new
from visual_control_utils.visualization import add_labels_to_image, add_arrow_prediction
import matplotlib.pyplot as plt



def evaluate_model(dataset_path, model_path):
    checkpoint_path = load_best_model(model_path)

    config_path = os.path.join(os.path.dirname(checkpoint_path), "..", "config.yaml")

    net_config = NetConfig(config_path)

    if net_config.dataset == "tfm":
        test_data, test_images = load_dataset(dataset_path, "Test", "test.json")
    else:
        test_data, test_images = load_dataset_new(dataset_path, net_config)

    # test_data = test_data[0:10]
    # test_images = test_images[0:10]


    test_dict = {
        "labels": test_data,
        "images": test_images
    }

    eval_transforms = A.Compose([
        A.LongestMaxSize(net_config.base_size, always_apply=True),
        A.PadIfNeeded(net_config.base_size, net_config.base_size, always_apply=True,
                      border_mode=cv2.BORDER_CONSTANT),
        A.Normalize(),
        ToTensorV2()
    ])

    model = VisualControl.load_from_checkpoint(checkpoint_path=checkpoint_path, dataset_path=dataset_path, lr=5e-2,
                                               base_size=net_config.base_size, batch_size=net_config.batch_size,
                                               net_config=net_config)

    # prints the learning_rate you used in this checkpoint

    device = "cuda:0"

    model.to(device)
    model.eval()
    model.freeze()

    results = {}

    visualize = False
    save = False
    output_dir = "visual_results"
    if save:
        if os.path.exists(output_dir):
            raise Exception("output dir already exists")
        os.mkdir(output_dir)

    w_mean, w_std, v_mean, v_std = net_config.norm_values

    for batch_idx in tqdm(range(0, len(test_dict["labels"]), net_config.batch_size)):
        images_batch = []
        labels = []
        raw_images = []
        for idx in range(batch_idx, min(batch_idx + net_config.batch_size, len(test_dict["labels"]))):
            image_path = os.path.join(dataset_path, "Test", "Images", test_dict["images"][idx])
            image = cv2.imread(image_path)
            raw_images.append(image)
            x = eval_transforms(image=image)
            y = net_config.encode(test_dict["labels"][idx])
            images_batch.append(x["image"])
            labels.append(y)

        predictions = model(default_collate(images_batch).to(device))
        for idx in range(0, len(predictions)):

            # d = encoder.get_smooth_estimation(predictions[idx].cpu().numpy())

            if net_config.head_type == NetConfig.CLASSIFICATION_TYPE:
                y_hat = from_logit_to_estimation(predictions[idx], net_config)
            else:
                y_hat = predictions[idx].cpu().numpy()
            y = labels[idx]

            y_hat[0] = (y_hat[0] * w_std) + w_mean
            y_hat[1] = (y_hat[1] * v_std) + v_mean

            y[0] = (y[0] * w_std) + w_mean
            y[1] = (y[1] * v_std) + v_mean

            results[idx + batch_idx] = {
                "label": y,
                "prediction": y_hat
            }



            if visualize or save:
                if visualize:
                    print("-----------------")
                    print(y)
                    print(y_hat)
                image_labels = add_labels_to_image(raw_images[idx], y, y_hat)

                if net_config.head_type == NetConfig.CLASSIFICATION_TYPE:
                    class_idx = from_one_hot_to_class(y_hat, net_config)
                    motors_info = net_config.get_real_values_from_estimation(class_idx)
                else:
                    motors_info = {"w": y_hat[0], "v": y_hat[1]}
                image_labels = add_arrow_prediction(image_labels, motors_info)

                if visualize:
                    cv2.imshow("Test", image_labels)
                    cv2.waitKey(0)
                if save:
                    output_file = os.path.join(output_dir, "{}.jpg".format(idx + batch_idx))
                    cv2.imwrite(output_file, image_labels)

    if net_config.head_type == NetConfig.CLASSIFICATION_TYPE:
        final_stats = {}
        gt = [results[x]["label"] for x in results]
        pred = [results[x]["prediction"] for x in results]
        acc = accuracy_score(gt, pred)
        acc2 = accuracy_score(gt, pred, normalize=False)
        print(acc)
        print("Global accuracy: {}/{}".format(acc2, len(gt)))
        final_stats["acc"] = acc
        final_stats["hits"] = int(acc2)
        final_stats["dataset_size"] = len(gt)

        for controller in net_config.classification_data:
            base_idx = 0
            if controller == "v" and "w" in net_config.classification_data:
                base_idx = len(net_config.classification_data['w']['classes'])

            indexes = net_config.softmax_config[controller]
            controller_labels = []
            for current_label in gt:
                controller_label = [current_label[i] for i in indexes].index(1.0)
                controller_labels.append(controller_label)

            controller_preds = []
            for controller_pred in pred:
                controller_preds.append([controller_pred[i] for i in indexes].index(1.0))

            acc1 = np.count_nonzero(np.abs(np.array(controller_preds) - np.array(controller_labels)) == 0)
            acc2 = np.count_nonzero(np.abs(np.array(controller_preds) - np.array(controller_labels)) <= 1)

            current_stats = {"acc1": acc1 / len(controller_preds),
                             "acc2": acc2 / len(controller_preds)}

            final_stats[controller] = current_stats

        out_json_stats = os.path.join(os.path.dirname(checkpoint_path), "..", "stats.json")

        json.dump(final_stats, open(out_json_stats, "w"), indent=4)

        # confusion matrices

        try:

            final_data = {

            }
            for current_data in results:
                gt = results[current_data]["label"]
                pred = results[current_data]["prediction"]
                class_data_gt = from_one_hot_to_class(gt, net_config)
                class_data_pred = from_one_hot_to_class(pred, net_config)

                for output_key in class_data_gt:
                    if output_key not in final_data:
                        final_data[output_key] = {
                            "label": [],
                            "prediction": []
                        }
                    final_data[output_key]["label"].append(class_data_gt[output_key])
                    final_data[output_key]["prediction"].append(class_data_pred[output_key])

            fig, axs = plt.subplots(1, 2)
            fig.set_size_inches(15, 7, forward=True)

            for idx, output_key in enumerate(final_data):
                ax = axs[idx]
                a = confusion_matrix(final_data[output_key]["label"], final_data[output_key]["prediction"],
                                     normalize="true", labels=net_config.get_str_labels()[output_key])
                plot_confusion_matrix(a, classes=net_config.get_str_labels()[output_key], title=output_key, ax=ax)

            plt.tight_layout()
            output_image_path = os.path.join(os.path.dirname(checkpoint_path), "..", "confusion_matrix.png")
            plt.savefig(output_image_path)
        except Exception as exc:
            print(Exception)

    else:
        stats = {}
        for result in results:
            current_result = results[result]
            for idx_controller, controller in enumerate(net_config.regression_data["controllers"]):
                current_diff = np.square(current_result["label"][idx_controller] - current_result["prediction"][idx_controller])
                if controller not in stats:
                    stats[controller] = []
                stats[controller].append(current_diff)

        final_stats = {}
        for controller in net_config.regression_data["controllers"]:
            final_stats[controller] = { "rmse":  float(np.square(np.mean(stats[controller]))) }

        out_json_stats = os.path.join(os.path.dirname(checkpoint_path), "..", "stats.json")
        json.dump(final_stats, open(out_json_stats, "w"), indent=4)

if __name__ == "__main__":
    dataset_path = "/home/frivas/Descargas/complete_dataset/"
    # checkpoint_path = "/home/frivas/devel/mio/github/2017-phd-francisco-rivas/deep_learning/python/networks/lightning_logs/version_0/checkpoints/rc-classification-epoch=12-val_acc=1.00-val_loss=0.05.ckpt"
    # checkpoint_path = "/home/frivas/devel/mio/github/2017-phd-francisco-rivas/deep_learning/python/networks/lightning_logs/version_1/checkpoints/rc-classification-epoch=34-rmse=0.83-val_loss=0.72.ckpt"
    # checkpoint_path = "/home/frivas/devel/mio/github/2017-phd-francisco-rivas/deep_learning/python/networks/lightning_logs/version_3/checkpoints/rc-classification-epoch=65-val_acc=1.00-val_loss=0.07.ckpt"
    # checkpoint_path = "/home/frivas/devel/mio/github/2017-phd-francisco-rivas/deep_learning/python/networks/lightning_logs/version_4/checkpoints/rc-classification-epoch=76-val_acc=0.95-val_loss=0.05.ckpt"
    # checkpoint_path = "/home/frivas/devel/mio/github/2017-phd-francisco-rivas/deep_learning/python/networks/lightning_logs/version_6/checkpoints/rc-classification-epoch=19-val_acc=1.00-val_loss=0.03.ckpt"
    # checkpoint_path = "/home/frivas/devel/mio/github/2017-phd-francisco-rivas/deep_learning/python/networks/lightning_logs/version_7/checkpoints/rc-classification-epoch=12-val_acc=1.00-val_loss=0.09.ckpt"
    # checkpoint_path = "/home/frivas/devel/mio/github/2017-phd-francisco-rivas/deep_learning/python/networks/lightning_logs/version_8/checkpoints/rc-classification-epoch=34-val_acc=0.98-val_loss=0.06.ckpt"
    # checkpoint_path = "/home/frivas/devel/mio/github/2017-phd-francisco-rivas/deep_learning/python/networks/lightning_logs/version_9/checkpoints/rc-classification-epoch=33-val_acc=0.98-val_loss=0.05.ckpt"


    model_path = "/home/frivas/devel/mio/github/2017-phd-francisco-rivas/deep_learning/python/networks/lightning_logs/version_14"

    evaluate_model(dataset_path, model_path)