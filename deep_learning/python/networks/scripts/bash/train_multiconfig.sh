#!/bin/bash

python train.py --config_file net_config/training_configurations/regression/mobile_large/exp1.yml
python train.py --config_file net_config/training_configurations/regression/mobile_large/exp2.yml
python train.py --config_file net_config/training_configurations/regression/mobile_large/exp3.yml


python train.py --config_file net_config/training_configurations/regression/mobile_small/exp1.yml
python train.py --config_file net_config/training_configurations/regression/mobile_small/exp2.yml
python train.py --config_file net_config/training_configurations/regression/mobile_small/exp3.yml

python train.py --config_file net_config/training_configurations/regression/pilot/exp1.yml
python train.py --config_file net_config/training_configurations/regression/pilot/exp2.yml
python train.py --config_file net_config/training_configurations/regression/pilot/exp3.yml

#sleep 10800
#python train.py --config_file net_config/training_configurations/7_4/large/exp1.yml
#python train.py --config_file net_config/training_configurations/7_4/large/exp2.yml
#python train.py --config_file net_config/training_configurations/7_4/large/exp3.yml
#python train.py --config_file net_config/training_configurations/7_4/large/exp4.yml
#python train.py --config_file net_config/training_configurations/7_4/large/exp5.yml
#python train.py --config_file net_config/training_configurations/7_4/large/exp6.yml
#python train.py --config_file net_config/training_configurations/7_4/large/exp7.yml
#
#python train.py --config_file net_config/training_configurations/7_4/small/exp1.yml
#python train.py --config_file net_config/training_configurations/7_4/small/exp2.yml
#python train.py --config_file net_config/training_configurations/7_4/small/exp3.yml
#python train.py --config_file net_config/training_configurations/7_4/small/exp4.yml
#python train.py --config_file net_config/training_configurations/7_4/small/exp5.yml
#python train.py --config_file net_config/training_configurations/7_4/small/exp6.yml
#python train.py --config_file net_config/training_configurations/7_4/small/exp7.yml

#python train.py --config_file net_config/training_configurations/8_9/exp1.yml
#python train.py --config_file net_config/training_configurations/8_9/exp2.yml
#python train.py --config_file net_config/training_configurations/8_9/exp3.yml
#python train.py --config_file net_config/training_configurations/8_9/exp4.yml
#python train.py --config_file net_config/training_configurations/8_9/exp5.yml
#python train.py --config_file net_config/training_configurations/8_9/exp6.yml
#python train.py --config_file net_config/training_configurations/8_9/exp7.yml


#python train.py --config_file net_config/SmallerVGG.yml



