import json
import os
import shutil
from random import shuffle

import cv2
import torchmetrics
from tqdm import tqdm
import random
import copy

import pytorch_lightning as pl
import torch
from pytorch_lightning.metrics.functional import accuracy, precision_recall
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision
import numpy as np

from net_config.net_config import NetConfig
from dataloaders.visual_control_simulated import VisualControlSimulated
from visual_control_utils.fix_json import read_malformed_json
from visual_control_utils.logits_conversion import from_logit_to_estimation, from_logit_to_estimation_class
from visual_control_utils.visual_datset_format import load_dataset
from visual_control_utils.visualization import add_labels_to_image, add_arrow_prediction
from models.backends.backends import get_backend_by_name


class VisualControl(pl.LightningModule):

    def __init__(self, dataset_path: str, lr: float, base_size: int,
                 batch_size: int, num_workers=6, pos_weight=True,
                 net_config:NetConfig=None):
        super().__init__()

        # Set our init args as class attributes
        self.dataset_path = dataset_path
        self.learning_rate = lr
        self.batch_size = batch_size
        self.base_size = base_size
        self.num_workers = num_workers
        self.pos_weight = pos_weight
        self.pos_weight_values = None
        self.net_config = net_config

        # Hardcode some dataset specific attributes
        self.num_classes = self.net_config.n_classes
        self.dims = (1, base_size, base_size)

        self.model = get_backend_by_name(self.net_config.backend)(pretrained=self.net_config.pretrained)

        if self.net_config.fc_head == "mobilenet_v3_large":
            self.model.classifier = nn.Sequential(
                nn.Linear(self.model.classifier[0].in_features, 1280),
                nn.Hardswish(inplace=True),
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(1280, self.num_classes),
            )
        elif self.net_config.fc_head == "pilotnet":
            self.model.classifier = nn.Sequential(
                nn.Linear(69696, 1164),
                nn.Linear(1164, 100),
                nn.Linear(100, 50),
                nn.Linear(50, 10),
                nn.Linear(10, self.num_classes)
            )

        # Define PyTorch model
        # resnet = False
        # if resnet:
        #     self.model = torchvision.models.resnet50(pretrained=True)
        #     self.model.fc = nn.Sequential(
        #         nn.Linear(self.model.fc.in_features, 1280),
        #         nn.Hardswish(inplace=True),
        #         nn.Dropout(p=0.5, inplace=True),
        #         nn.Linear(1280, self.num_classes),
        #     )
        # else:
        #     self.model = torchvision.models.mobilenet_v3_large(pretrained=True)
        #     self.model.classifier = nn.Sequential(
        #         nn.Linear(self.model.classifier[0].in_features, 1280),
        #         nn.Hardswish(inplace=True),
        #         nn.Dropout(p=0.2, inplace=True),
        #         nn.Linear(1280, self.num_classes),
        #     )

    def forward(self, x):
        x = self.model(x)
        if self.net_config.head_type == NetConfig.CLASSIFICATION_TYPE:
            x = torch.sigmoid(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x["image"])

        if self.net_config.head_type == NetConfig.CLASSIFICATION_TYPE:
            if self.pos_weight_values is not None:
                loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(self.pos_weight_values).to(self.device))
            else:
                loss_fn = nn.BCEWithLogitsLoss()
        elif self.net_config.head_type == NetConfig.REGRESSION_TYPE:
            loss_fn = nn.MSELoss()
        else:
            raise Exception("NetConfig head type: {} not supported".format(self.net_config.head_type))

        loss = loss_fn(logits, y)
        self.log('train_loss', loss)
        self.log('lr', self.learning_rate)

        return loss

    def validation_step(self, batch, batch_idx):

        x, y = batch
        logits = self.model(x["image"])

        if self.net_config.head_type == NetConfig.CLASSIFICATION_TYPE:
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, y)
            acc = []
            for output_key in ["w", "v"]:
                current_estimations = []
                gt = []
                if output_key in self.net_config.softmax_config:
                    for idx, pred in enumerate(logits):
                        class_idx = from_logit_to_estimation_class(pred, self.net_config)[output_key]
                        current_estimations.append(class_idx)
                        current_gt = y[idx].cpu().numpy()
                        current_gt = [current_gt[i] for i in self.net_config.softmax_config[output_key]]
                        gt.append(np.argmax(current_gt, axis=0))

                preds = torch.IntTensor(current_estimations).to(self.device)
                gt = torch.IntTensor(gt).to(self.device)
                key_acc = torchmetrics.functional.accuracy(preds, gt)
                self.log('val_acc_{}'.format(output_key), key_acc, prog_bar=True)
                acc.append(key_acc)

            acc = torch.mean(torch.FloatTensor(acc))
            if batch_idx == 0:
                images_to_show = []
                for i in range (0, 10):
                    image = x["image"][i].to("cpu").numpy().transpose([1, 2, 0])
                    gt = y[i].to("cpu").numpy()
                    # current_prediction = torch.sigmoid(logits[i].to("cpu").detach())
                    current_prediction = logits[i].to("cpu").detach()

                    one_hot = from_logit_to_estimation(current_prediction, self.net_config)

                    image *= (0.229, 0.224, 0.225)
                    image += (0.485, 0.456, 0.406)
                    image *= 255
                    image = np.ascontiguousarray(image, dtype=np.uint8)

                    image = add_labels_to_image(image, gt, one_hot)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    images_to_show.append(image)

                sample_imgs = torch.from_numpy((np.array(images_to_show)).transpose((0,3,1,2))).to(self.device)
                grid = torchvision.utils.make_grid(sample_imgs, nrow=5)
                self.logger.experiment.add_image('generated_images', grid, self.current_epoch)
                self.log('val_acc', acc, prog_bar=True)
        elif self.net_config.head_type == NetConfig.REGRESSION_TYPE:
            loss_fn = nn.MSELoss()
            loss = loss_fn(logits, y)
            self.log('rmse', torch.sqrt(loss), prog_bar=True)
            if batch_idx == 0:
                images_to_show = []
                for i in range(0, min(10, len(x["image"]))):
                    image = x["image"][i].to("cpu").numpy().transpose([1, 2, 0])
                    gt = y[i].to("cpu").numpy()
                    # current_prediction = torch.sigmoid(logits[i].to("cpu").detach())
                    current_prediction = logits[i].to("cpu").detach()

                    image *= (0.229, 0.224, 0.225)
                    image += (0.485, 0.456, 0.406)
                    image *= 255
                    image = np.ascontiguousarray(image, dtype=np.uint8)

                    image = add_labels_to_image(image, gt, current_prediction.numpy())
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    images_to_show.append(image)

                sample_imgs = torch.from_numpy((np.array(images_to_show)).transpose((0, 3, 1, 2))).to(self.device)
                grid = torchvision.utils.make_grid(sample_imgs, nrow=5)
                self.logger.experiment.add_image('generated_images', grid, self.current_epoch)
        else:
            raise Exception("NetConfig head type: {} not supported".format(self.net_config.head_type))

        self.log('val_loss', loss, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        # optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, weight_decay=0.9)
        optimizer = torch.optim.Adam(self.parameters())
        # self.exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
        return [optimizer] \
            # , [self.exp_lr_scheduler]

    ####################
    # DATA RELATED HOOKS
    ####################

    def duplicate_with_vertical_flip(self, dataset):
        duplicate_samples = True
        if duplicate_samples:
            extra_samples = {"labels": [],
                             "images": [],
                             "vertical_flip": []}
            dataset["vertical_flip"] = []
            for idx, sample in enumerate(dataset["labels"]):
                dataset["vertical_flip"].append(0)
                new_label = copy.copy(sample)
                new_label["w"] = -new_label["w"]
                extra_samples["vertical_flip"].append(1)
                extra_samples["labels"].append(new_label)
                extra_samples["images"].append(dataset["images"][idx])

            dataset["labels"] += extra_samples["labels"]
            dataset["images"] += extra_samples["images"]
            dataset["vertical_flip"] += extra_samples["vertical_flip"]

            #shufle
            c = list(zip(dataset["labels"], dataset["images"], dataset["vertical_flip"]))
            random.shuffle(c)
            dataset["labels"], dataset["images"], dataset["vertical_flip"] = zip(*c)

        return dataset



    def prepare_data(self):
        train_data, train_images = load_dataset(self.dataset_path, "Train", "train.json")
        c = list(zip(train_data, train_images))
        random.shuffle(c)
        train_data, train_images = zip(*c)
        train_data = list(train_data)
        train_images = list(train_images)

        cut_valid_idx = int(len(train_data) * 0.8)

        self.train_dict = {
            "labels": train_data[0:cut_valid_idx],
            "images": train_images[0:cut_valid_idx],
        }

        self.valid_dict = {
            "labels": train_data[cut_valid_idx:],
            "images": train_images[cut_valid_idx:],
        }

        self.train_dict = self.duplicate_with_vertical_flip(self.train_dict)
        self.valid_dict = self.duplicate_with_vertical_flip(self.valid_dict)


        test_data, test_images = load_dataset(self.dataset_path, "Test", "test.json")

        self.test_dict = {
            "labels": test_data,
            "images": test_images
        }

    def setup(self, stage=None):

        #copy network config
        os.makedirs(self.logger.log_dir)
        shutil.copy(self.net_config.config_file, os.path.join(self.logger.log_dir, "config.yaml"))

        if self.net_config.head_type != NetConfig.CLASSIFICATION_TYPE:
            self.pos_weight = None
        try:
            if self.pos_weight:
                val_set = VisualControlSimulated(os.path.join(self.dataset_path, "Train"), self.valid_dict,
                                         self.base_size, self.net_config, split='val')

                ones_count = np.zeros(self.num_classes)
                for i in tqdm(range(0, len(val_set))):
                    _, target = val_set.__getitem__(i)
                    ones_count += target
                zero_count = np.full((self.num_classes,), len(val_set), dtype=np.float64)
                zero_count -= ones_count

                self.pos_weight_values = zero_count / ones_count
                self.pos_weight_values[self.pos_weight_values == np.inf] = 0
                print(self.pos_weight_values)

        except Exception as exc:
            pass

    def train_dataloader(self):
        train_set = VisualControlSimulated(os.path.join(self.dataset_path, "Train"), self.train_dict,
                                           self.base_size, self.net_config, split='train')
        train_loader = DataLoader(train_set, batch_size=self.batch_size,
                                  num_workers=self.num_workers)

        return train_loader

    def val_dataloader(self):
        val_set = VisualControlSimulated(os.path.join(self.dataset_path, "Train"), self.valid_dict,
                                         self.base_size, self.net_config, split='val')
        valid_loader = DataLoader(val_set, batch_size=self.batch_size,
                                  num_workers=self.num_workers)
        return valid_loader

    def test_dataloader(self):
        # return DataLoader(self.mnist_test, batch_size=32)
        pass
