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
    
gallery1:
  - url: /assets/images/classification/mobilenet/WV_3_2/w-v/w/confusion_matrix.png
    image_path: /assets/images/classification/mobilenet/WV_3_2/w-v/w/confusion_matrix.png
    alt: "WV_3_2_Confusion Matrix"
    
gallery2:
  - url: /assets/images/classification/mobilenet/WV_3_2/w-v/v/confusion_matrix.png
    image_path: /assets/images/classification/mobilenet/WV_3_2/w-v/v/confusion_matrix.png
    alt: "WV_3_2_Confusion Matrix"
    
gallery3:
  - url: /assets/images/classification/mobilenet/WV_7_4/w+v/confusion_matrix.png
    image_path: /assets/images/classification/mobilenet/WV_7_4/w+v/confusion_matrix.png
    alt: "WV_3_2_Confusion Matrix"
    
gallery4:
  - url: /assets/images/classification/mobilenet/WV_7_4/w-v/w/confusion_matrix.png
    image_path: /assets/images/classification/mobilenet/WV_7_4/w-v/w/confusion_matrix.png
    alt: "WV_3_2_Confusion Matrix"
    
gallery5:
  - url: /assets/images/classification/mobilenet/WV_3_2/w-v/v/confusion_matrix.png
    image_path: /assets/images/classification/mobilenet/WV_7_4/w-v/v/confusion_matrix.png
    alt: "WV_3_2_Confusion Matrix"
---


With these sets of experiments we want to evaluate if a classification network is enough to take the driving control of a f1 model which should complete rounds in several circuits following a line. 

## WV_3_2
In this experiment we want to evaluate the following configuration:
* 3 classes for the w
* 2 classes for the v
[Configuration file](https://github.com/RoboticsLabURJC/2017-phd-francisco-rivas/blob/master/deep_learning/python/networks/net_config/WV_3_2_CLASSES.yml)


### Single network

| w+v Accuracy | w Accuracy | v Accuracy | w Accuracy (top2) | v Accuracy (top2) |
|:-------:|:--------:|:--------:|:--------:|:--------:|
| 0.9554| 0.976  | 0.972   | 0.998 | 1.0 |

Confusion matrix:
{% include gallery id="gallery" caption="" %}


### Network per controller


| w+v Accuracy | w Accuracy | v Accuracy | w Accuracy (top2) | v Accuracy (top2) |
|:-------:|:--------:|:--------:|:--------:|:--------:|
| N/A | 0.9778  |  0.971  | 0.999 | 1.0 |

Confusion matrix:
{% include gallery id="gallery1" caption="" %}
{% include gallery id="gallery2" caption="" %}




## WV_7_4
In this experiment we want to evaluate the following configuration:
* 7 classes for the w
* 4 classes for the v
[Configuration file](https://github.com/RoboticsLabURJC/2017-phd-francisco-rivas/blob/master/deep_learning/python/networks/net_config/WV_7_4_CLASSES.yml)

### Single network

| w+v Accuracy | w Accuracy | v Accuracy | w Accuracy (top2) | v Accuracy (top2) |
|:-------:|:--------:|:--------:|:--------:|:--------:|
| 0.932| 0.954  | 0.970   | 0.999 | 972 |

Confusion matrix:
{% include gallery id="gallery3" caption="" %}


### Network per controller


| w+v Accuracy | w Accuracy | v Accuracy | w Accuracy (top2) | v Accuracy (top2) |
|:-------:|:--------:|:--------:|:--------:|:--------:|
| N/A | 0.956 | 0.969 | 0.999 | 0.971 |

Confusion matrix:
{% include gallery id="gallery4" caption="" %}
{% include gallery id="gallery5" caption="" %}



## Summary
| Configuration | w Accuracy - RMSE | v Accuracy - RMSE | SimpleCircuit | Curves | Nurburgring | Montmelo |
|:-------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| WV_3_2 (w + v) | 0.976  | 0.972 | 66 | 130 | N/A - | N/A - |  
