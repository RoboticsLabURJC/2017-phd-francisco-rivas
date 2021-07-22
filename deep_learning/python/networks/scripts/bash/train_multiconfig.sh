#!/bin/bash



python train.py --config_file net_config/resnet/34/WV_3_2_CLASSES.yml
python train.py --config_file net_config/resnet/34/WV_7_4_CLASSES.yml
python train.py --config_file net_config/resnet/34/WV_7_5_CLASSES.yml
python train.py --config_file net_config/resnet/34/WV_9_4_CLASSES.yml
python train.py --config_file net_config/resnet/34/WV_12_8_CLASSES.yml
#
python train.py --config_file net_config/resnet/34/v/WV_3_2_CLASSES.yml
python train.py --config_file net_config/resnet/34/w/WV_3_2_CLASSES.yml
#
python train.py --config_file net_config/resnet/34/v/WV_7_4_CLASSES.yml
python train.py --config_file net_config/resnet/34/w/WV_7_4_CLASSES.yml

python train.py --config_file net_config/resnet/34/v/WV_7_5_CLASSES.yml
python train.py --config_file net_config/resnet/34/w/WV_7_5_CLASSES.yml

python train.py --config_file net_config/resnet/34/v/WV_9_4_CLASSES.yml
python train.py --config_file net_config/resnet/34/w/WV_9_4_CLASSES.yml

python train.py --config_file net_config/resnet/34/v/WV_12_8_CLASSES.yml
python train.py --config_file net_config/resnet/34/w/WV_12_8_CLASSES.yml

python train.py --config_file net_config/resnet/34/ResnetRegression.yml

python train.py --config_file net_config/resnet/34/ResnetRegressionV.yml
python train.py --config_file net_config/resnet/34/ResnetRegressionW.yml


#
#python train.py --config_file net_config/vgg/16/WV_3_2_CLASSES.yml
#python train.py --config_file net_config/vgg/16/WV_7_4_CLASSES.yml
#python train.py --config_file net_config/vgg/16/WV_7_5_CLASSES.yml
#python train.py --config_file net_config/vgg/16/WV_9_4_CLASSES.yml
#python train.py --config_file net_config/vgg/16/WV_12_8_CLASSES.yml
#
#python train.py --config_file net_config/vgg/16/v/WV_3_2_CLASSES.yml
#python train.py --config_file net_config/vgg/16/w/WV_3_2_CLASSES.yml
#
#python train.py --config_file net_config/vgg/16/v/WV_7_4_CLASSES.yml
#python train.py --config_file net_config/vgg/16/w/WV_7_4_CLASSES.yml
#
#python train.py --config_file net_config/vgg/16/v/WV_7_5_CLASSES.yml
#python train.py --config_file net_config/vgg/16/w/WV_7_5_CLASSES.yml
#
#python train.py --config_file net_config/vgg/16/v/WV_9_4_CLASSES.yml
#python train.py --config_file net_config/vgg/16/w/WV_9_4_CLASSES.yml
#
#python train.py --config_file net_config/vgg/16/v/WV_12_8_CLASSES.yml
#python train.py --config_file net_config/vgg/16/w/WV_12_8_CLASSES.yml
#
#python train.py --config_file net_config/vgg/16/VGGRegression.yml
#python train.py --config_file net_config/MobileSmallRegressionCustom.yml

