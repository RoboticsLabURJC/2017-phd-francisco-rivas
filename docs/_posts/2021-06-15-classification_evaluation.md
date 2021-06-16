---
title: "Classification Networks - Analysis"
excerpt: "Analysis os multiple configuration of classification networks."

sidebar:
  nav: "docs"

categories:
- Deep Learning
tags:
- PyTorch
- Classification
- Autonomous driving

author: Francisco Rivas
pinned: false


gallery:
  - url: /assets/images/classification/mobilenet/WV_3_2/w+v/confusion_matrix.png
    image_path: /assets/images/classification/mobilenet/WV_3_2/w+v/confusion_matrix.png
    alt: "WV_3_2_Confusion Matrix"
    

---


With these sets of experiments we want to evaluate if a classification network is enough to take the driving control of a f1 model which should complete rounds in several circuits following a line. 

## WV_3_2
In this experiment we want to evaluate the following configuration:
* 3 classes for the w
* 2 classes for the v

[Configuration file](https://github.com/RoboticsLabURJC/2017-phd-francisco-rivas/blob/master/deep_learning/python/networks/net_config/WV_3_2_CLASSES.yml)


| w+v Accuracy | w Accuracy | v Accuracy | w Accuracy (top2) | v Accuracy (top2) |
|:-------:|:--------:|:--------:|:--------:|:--------:|
| 0.9554| 0.976  | 0.972   | 0.998 | 1.0 |

Confusion matrix:
{% include gallery id="gallery" caption="" %}



