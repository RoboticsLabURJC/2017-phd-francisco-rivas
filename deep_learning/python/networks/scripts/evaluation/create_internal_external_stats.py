import os
from builtins import enumerate

import yaml
import json
from networks.evaluation import evaluate_model
import pandas as pd

if __name__ == "__main__":
    out_df = pd.DataFrame()
    dataset_path = "/home/frivas/Descargas/complete_dataset/"
    dataset_path = "/media/nas/PHD/datasets_opencv"

    multiple_evaluation_file = "/home/frivas/devel/mio/github/BehaviorMetrics/behavior_metrics/configs/default-multiple_NEW.yml"
    multiple_evaluation_file = "/home/frivas/devel/mio/github/BehaviorMetrics/behavior_metrics/configs/default-multiple_tfm.yml"

    # stats_file = "/home/frivas/devel/mio/github/BehaviorMetrics/behavior_metrics/bag_analysis/stats.json"

    input_data = yaml.load(open(multiple_evaluation_file), Loader=yaml.FullLoader)

    for element in input_data['Behaviors']['Robot']['Parameters']['Model']:
        internal_stats = {}
        paths_to_evaluate = []
        if isinstance(element, list):
            paths_to_evaluate += element
        else:
            paths_to_evaluate.append(element)

        for path_to_evaluate in paths_to_evaluate:
            current_stats_file = os.path.join(path_to_evaluate, "stats.json")
            if True: #not os.path.exists(current_stats_file):
                print("Evaluating: {}".format(path_to_evaluate))
                evaluate_model(dataset_path, path_to_evaluate)
            else:
                print("Using evaluation cache for: {}".format(path_to_evaluate))



    # external_stats = json.load(open(stats_file))
    #
    # for idx_run, element in enumerate(input_data['Behaviors']['Robot']['Parameters']['Model']):
    #     #get model_info
    #     multi_model = isinstance(element, list)
    #     if multi_model:
    #         current_model_config_file = os.path.join(element[0], "config.yaml")
    #     else:
    #         current_model_config_file = os.path.join(element, "config.yaml")
    #     current_model_config = yaml.load(open(current_model_config_file), Loader=yaml.FullLoader)
    #     exp_name = list(current_model_config.keys())[0]
    #     mode = current_model_config[exp_name]["mode"]
    #     model = current_model_config[exp_name]["backend"]
    #     head = current_model_config[exp_name]["fc_head"]
    #     base_size = current_model_config[exp_name]["base_size"]
    #     batch_size = current_model_config[exp_name]["batch_size"]
    #     normalize_input = current_model_config[exp_name]["normalize_input"]
    #
    #     non_common_samples_mult_factor = current_model_config[exp_name].get("non_common_samples_mult_factor", 0)
    #
    #
    #     len_v = 1
    #     len_w = 1
    #     if mode == "classification":
    #         if multi_model:
    #             len_w = len(current_model_config[exp_name]['classification_data']['w'])
    #             v_config_file = os.path.join(element[1], "config.yaml")
    #             v_config = yaml.load(open(v_config_file), Loader=yaml.FullLoader)
    #             v_exp_name = list(v_config.keys())[0]
    #             len_v = len(v_config[v_exp_name]['classification_data']['v'])
    #         else:
    #             len_w = len(current_model_config[exp_name]['classification_data']['w'])
    #             len_v = len(current_model_config[exp_name]['classification_data']['v'])
    #
    #
    #     # get internal_stats
    #
    #     acc = {"v":0, "w":0}
    #
    #     paths_to_evaluate = []
    #     if isinstance(element, list):
    #         paths_to_evaluate += element
    #     else:
    #         paths_to_evaluate.append(element)
    #
    #     for path_to_evaluate in paths_to_evaluate:
    #         current_stats_file = os.path.join(path_to_evaluate, "stats.json")
    #         current_stats_data = json.load(open(current_stats_file))
    #
    #         for controller in ["w", "v"]:
    #             if controller in current_stats_data:
    #                 if mode == "classification":
    #                     acc[controller] = current_stats_data[controller]["acc1"]
    #                 else:
    #                     acc[controller] = current_stats_data[controller]["rmse"]
    #     df_idx = len(out_df)
    #     for circuit in external_stats:
    #         out_df.loc[df_idx, "exp_name"] = exp_name
    #         out_df.loc[df_idx, "mode"] = mode
    #         out_df.loc[df_idx, "len_v"] = len_v
    #         out_df.loc[df_idx, "len_w"] = len_w
    #
    #         out_df.loc[df_idx, "model"] = model
    #         out_df.loc[df_idx, "head"] = head
    #         out_df.loc[df_idx, "base_size"] = base_size
    #         out_df.loc[df_idx, "batch_size"] = batch_size
    #         out_df.loc[df_idx, "normalize_input"] = normalize_input
    #         out_df.loc[df_idx, "non_common_samples_mult_factor"] = non_common_samples_mult_factor
    #
    #         out_df.loc[df_idx, "acc_v"] = acc["v"]
    #         out_df.loc[df_idx, "acc_w"] = acc["w"]
    #
    #
    #
    #         for idx_sample, sample_count in enumerate(range(idx_run*2, idx_run * 2 + 2)):
    #             out_df.loc[df_idx, circuit + "_{}_Percent".format(idx_sample)] =  external_stats[circuit]["percentage_completed"][sample_count]
    #             out_df.loc[df_idx, circuit + "_{}_Time".format(idx_sample)] =  external_stats[circuit]["lap_seconds"][sample_count]
    #
    # out_df.to_csv("stats.csv")