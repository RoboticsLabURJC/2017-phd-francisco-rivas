from __future__ import print_function, division
import random
import json
import os
import copy

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from dataloaders.visual_control_encoder import VisualControlEncoder


class VisualControlSimulated(Dataset):

    def __init__(self,
                 base_dir: str,
                 dataset: dict,
                 base_size: int,
                 encoder: VisualControlEncoder,
                 split='train'
                 ):
        """
        :param base_dir:
        :param split: train and val will come in from caller
        :param transform: transform to apply
        """
        super().__init__()
        self.base_size = base_size
        self._base_dir = base_dir
        self.dataset = dataset

        self.encoder = encoder
        self.N_CLASSES = self.encoder.n_classes

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.dataset["labels"])))

        self.tr_transform = self.compose_transform_tr()
        self.val_transform = self.compose_transform_val()

    def __len__(self):
        return len(self.dataset["labels"])

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)

        for split in self.split:
            if split == "train":
                return self.tr_transform(image=_img), _target
            elif split == 'val':
                return self.val_transform(image=_img), _target

    def _make_img_gt_point_pair(self, index):

        image_filename = self.dataset["images"][index]
        raw_label = self.dataset["labels"][index]
        image = cv2.imread(os.path.join(self._base_dir, "Images", image_filename))
        if "vertical_flip" in self.dataset:
            if self.dataset["vertical_flip"][index]:
                image = cv2.flip(image, 1)
                # cv2.imshow("test", image)
                # cv2.waitKey(0)
        label = self.encoder.encode(raw_label)
        return image, label


    def compose_transform_tr(self):
        composed_transforms = A.Compose([
            # A.HorizontalFlip(),
            # A.Rotate(),
            A.RandomScale(0.2),
            A.LongestMaxSize(self.base_size, always_apply=True),
            A.PadIfNeeded(self.base_size, self.base_size, always_apply=True,
                          border_mode=cv2.BORDER_CONSTANT),
            # A.RandomSizedCrop((self.args.crop_size, self.args.crop_size),height=self.args.crop_size, width=self.args.crop_size),
            A.RandomBrightnessContrast(),
            A.HueSaturationValue(),
            A.FancyPCA(),
            A.RandomGamma(),
            A.GaussianBlur(),
            A.Normalize(),
            ToTensorV2()
        ]
        )

        return composed_transforms

    def compose_transform_val(self):
        composed_transforms = A.Compose([
            A.LongestMaxSize(self.base_size, always_apply=True),
            A.PadIfNeeded(self.base_size, self.base_size, always_apply=True,
                          border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(),
            ToTensorV2()
        ])

        return composed_transforms

    def __str__(self):
        return 'VisualControl(split=' + str(self.split) + ')'
