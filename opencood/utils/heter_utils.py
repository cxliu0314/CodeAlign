# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib
# Agent Selection Module for Heterogeneous Collaboration.

import numpy as np
import random
import os
from collections import OrderedDict
import json

class Adaptor:
    def __init__(self, 
                ego_modality, 
                model_modality_list, 
                modality_assignment,
                lidar_channels_dict,
                mapping_dict,
                cav_preference,
                train):
        self.ego_modality = ego_modality
        self.model_modality_list = model_modality_list
        self.modality_assignment = modality_assignment
        self.lidar_channels_dict = lidar_channels_dict
        self.mapping_dict = mapping_dict
        if cav_preference is None:
            cav_preference = dict.fromkeys(model_modality_list, 1/len(model_modality_list))
        self.cav_preferece = cav_preference # training, probability for setting non-ego cav modality
        self.train = train


    def reorder_cav_list(self, cav_list, scenario_name):
        """
        When evaluation, make the cav that could be ego modality after mapping be the first.

        This can check the training effect of aligner.

        work in basedataset -> reinitialize
        """
        if self.train:
            # shuffle the cav list
            random.shuffle(cav_list)
            return cav_list

        assignment = self.modality_assignment[scenario_name]
        if assignment[cav_list[0]] not in self.ego_modality:
            ego_cav = None
            for cav_id, modality in assignment.items():
                if self.mapping_dict[modality] in self.ego_modality: # after mapping the modality is ego
                    ego_cav = cav_id
                    break

            if ego_cav is None:
                return cav_list

            other_cav = sorted(list(assignment.keys()))
            other_cav.remove(ego_cav)
            cav_list = [ego_cav] + other_cav

        return cav_list
    
    def reassign_cav_modality(self, modality_name, idx_in_cav_list,scenario_name = None):
        """
        work in basedataset -> reinitialize
        """
        if self.train: 
            # always assign the ego_modality to idx 0 in cav_list
            if idx_in_cav_list == 0:
                return np.random.choice(self.ego_modality.split("&"))
            return random.choices(list(self.cav_preferece.keys()), weights=self.cav_preferece.values())[0]
        else:
            return self.mapping_dict[modality_name]

    def unmatched_modality(self, cav_modality):
        """
        work in 
            intermediate_heter_fusion_dataset -> __getitem__
            late_heter_fusion_dataset -> get_item_test

        Returns:
            True/False. If the input modality is in the model_modality_list
        """
        return cav_modality not in self.model_modality_list

    def switch_lidar_channels(self, cav_modality, lidar_file_path):
        """
        Currently only support OPV2V
        """
        if self.lidar_channels_dict.get(cav_modality, None) == 32:
            return lidar_file_path.replace("OPV2V","OPV2V_Hetero").replace(".pcd", "_32.pcd")
        if self.lidar_channels_dict.get(cav_modality, None) == 16:
            return lidar_file_path.replace("OPV2V","OPV2V_Hetero").replace(".pcd", "_16.pcd")
        return lidar_file_path


class Adaptor_group:
    def __init__(self, 
                ego_modality, 
                model_modality_list, 
                modality_assignment,
                lidar_channels_dict,
                mapping_dict,
                cav_preference,
                train, 
                params):
        self.ego_modality = ego_modality
        self.model_modality_list = model_modality_list
        self.modality_assignment = modality_assignment
        self.lidar_channels_dict = lidar_channels_dict
        self.mapping_dict = mapping_dict
        if cav_preference is None:
            cav_preference = dict.fromkeys(model_modality_list, 1/len(model_modality_list))
        self.cav_preferece = cav_preference # training, probability for setting non-ego cav modality
        self.train = train
        self.val = params['val'] if 'val' in params else False
        self.params = params
        random.seed(303)

        if 'plus' in params["heter"]["assignment_path"]:
            self.opv2vplus = True
        else:
            self.opv2vplus = False

        if self.train:
            self.divide_scene()

    def divide_scene(self):
        # rearrange assignment divided by hetero group
        self.heter_group = self.params['heter']['heter_group']
        self.group_num = len(self.heter_group)
        self.assigned_scene_group = {}
        i = 0
        for scene in self.modality_assignment.keys():
            self.assigned_scene_group[scene] = i
            i = (i+1)%self.group_num
        
        new_assignment = {}
        for scene, cav_dict in self.modality_assignment.items():
            group_id = self.assigned_scene_group[scene]
            available_modalities = self.heter_group[group_id]
            cav_ids = list(cav_dict.keys())
            random.shuffle(available_modalities)

            modality_list = []
            for i in range(len(cav_ids)):
                modality = available_modalities[i % len(available_modalities)]
                modality_list.append(modality)
            random.shuffle(modality_list)

            new_assignment[scene] = {}
            for cav_id, modality in zip(cav_ids, modality_list):
                new_assignment[scene][cav_id] = modality
        self.modality_assignment = new_assignment
        return new_assignment

    def reorder_cav_list(self, cav_list, scenario_name):
        """
        When evaluation, make the cav that could be ego modality after mapping be the first.

        This can check the training effect of aligner.

        work in basedataset -> reinitialize
        """
        if self.train:
            # shuffle the cav list
            random.shuffle(cav_list)
            return cav_list

        assignment = self.modality_assignment[scenario_name]
        if assignment[cav_list[0]] not in self.ego_modality:
            ego_cav = None
            for cav_id, modality in assignment.items():
                if self.mapping_dict[modality] in self.ego_modality: # after mapping the modality is ego
                    ego_cav = cav_id
                    break

            if ego_cav is None:
                return cav_list

            other_cav = sorted(list(assignment.keys()))
            other_cav.remove(ego_cav)
            cav_list = [ego_cav] + other_cav

        return cav_list
    
    def reassign_cav_modality(self, modality_name, idx_in_cav_list, scenario_name):
        """
        work in basedataset -> reinitialize
        """
        if self.train: 
            group_id = self.assigned_scene_group[scenario_name]
            available_modalities = self.heter_group[group_id]
            return random.choice(available_modalities)
        elif self.val:
            available_modalities = self.ego_modality
            return random.choice(available_modalities)
        else:
            return self.mapping_dict[modality_name]

    def unmatched_modality(self, cav_modality):
        """
        work in 
            intermediate_heter_fusion_dataset -> __getitem__
            late_heter_fusion_dataset -> get_item_test

        Returns:
            True/False. If the input modality is in the model_modality_list
        """
        return cav_modality not in self.model_modality_list

    def switch_lidar_channels(self, cav_modality, lidar_file_path):
        """
        Currently only support OPV2V
        """
        if self.opv2vplus:
            if self.lidar_channels_dict.get(cav_modality, None) == 32:
                return lidar_file_path.replace("OPV2VH_MoreAgents","OPV2VH_MoreAgents_LiDAR").replace("test","test_32")
            if self.lidar_channels_dict.get(cav_modality, None) == 16:
                raise ValueError("channel 16 is not in opv2vplus.")
                return lidar_file_path.replace("OPV2VH_MoreAgents","OPV2VH_MoreAgents_LiDAR").replace("test","test_16")
            if self.lidar_channels_dict.get(cav_modality, None) is None: # empty dict
                return lidar_file_path.replace("OPV2VH_MoreAgents","OPV2VH_MoreAgents_LiDAR").replace("test","test_64")
        else:
            if self.lidar_channels_dict.get(cav_modality, None) == 32:
                return lidar_file_path.replace("OPV2V","OPV2V_Hetero").replace(".pcd", "_32.pcd")
            if self.lidar_channels_dict.get(cav_modality, None) == 16:
                return lidar_file_path.replace("OPV2V","OPV2V_Hetero").replace(".pcd", "_16.pcd")
        return lidar_file_path


def assign_modality_4(root_dir="dataset/OPV2V", output_path="opencood/modality_assign/opv2v_4modality.json"):
    np.random.seed(303)
    splits = ['train', 'test', 'validate']
    scenario_cav_modality_dict = OrderedDict()

    for split in splits:
        split_path = os.path.join(root_dir, split)
        scenario_folders = sorted([os.path.join(split_path, x)
                                    for x in os.listdir(split_path) if
                                    os.path.isdir(os.path.join(split_path, x))])

        for scenario_folder in scenario_folders:
            scenario_name = scenario_folder.split('/')[-1]
            scenario_cav_modality_dict[scenario_name] = OrderedDict()

            cav_list = sorted([x for x in os.listdir(scenario_folder) \
                                if os.path.isdir(os.path.join(scenario_folder, x))])

            perm = np.random.permutation(4) + 1 
            for j, cav_id in enumerate(cav_list):
                scenario_cav_modality_dict[scenario_name][cav_id] = 'm'+str(perm[j%4]) # m1 or m2 or m3 or m4

    
    with open(output_path, "w") as f:
        json.dump(scenario_cav_modality_dict, f, indent=4, sort_keys=True)


def assign_modality_4_in_order(root_dir="dataset/OPV2V", output_path="opencood/modality_assign/opv2v_4modality_in_order.json"):
    """
        We assign each cav with a modality in order. Use m1m2m3m4 circularly
        cav1 -> m1
        cav2 -> m2
        cav3 -> m3
        cav4 -> m4
        cav5 -> m1
        cav6 -> m2
        ...
    """
    splits = ['test']
    scenario_cav_modality_dict = OrderedDict()

    for split in splits:
        split_path = os.path.join(root_dir, split)
        scenario_folders = sorted([os.path.join(split_path, x)
                                    for x in os.listdir(split_path) if
                                    os.path.isdir(os.path.join(split_path, x))])

        for scenario_folder in scenario_folders:
            scenario_name = scenario_folder.split('/')[-1]
            scenario_cav_modality_dict[scenario_name] = OrderedDict()

            cav_list = sorted([x for x in os.listdir(scenario_folder) \
                                if os.path.isdir(os.path.join(scenario_folder, x))])
            if cav_list[0] == '-1':
                cav_list = cav_list[1:] + cav_list[:1]

            for j, cav_id in enumerate(cav_list):
                scenario_cav_modality_dict[scenario_name][cav_id] = 'm'+str(j%4+1)

    
    with open(output_path, "w") as f:
        # if V2XSet, you can set sort_keys to False
        json.dump(scenario_cav_modality_dict, f, indent=4, sort_keys=True)

if __name__ == "__main__":
    # assign_modality()
    # assign_modality_4_in_order()
    assign_modality_4('dataset/V2XSET', output_path='opencood/modality_assign/v2xset_4modality.json')
