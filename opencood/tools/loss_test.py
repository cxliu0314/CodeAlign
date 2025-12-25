# -*- coding: utf-8 -*-
import argparse
import os
import json
from typing import OrderedDict
import torch
import torchvision
from torch.utils.data import DataLoader
import numpy as np
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
from opencood.visualization import vis_utils, my_vis, simple_vis_stamp
from opencood.utils.common_utils import update_dict
from matplotlib import pyplot as plt
from tqdm import tqdm
import cv2
import numpy as np

torch.multiprocessing.set_sharing_strategy("file_system")
# CUDA_VISIBLE_DEVICES=3 python opencood/tools/feature_vis.py --model_dir 

def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("model_dir", type=str, help="Continued training path")
    parser.add_argument(
        "--fusion_method", type=str, default="intermediate", help="no, no_w_uncertainty, late, early or intermediate"
    )
    parser.add_argument("--save_vis_interval", type=int, default=40, help="interval of saving visualization")
    parser.add_argument(
        "--save_npy", action="store_true", help="whether to save prediction and gt result" "in npy file"
    )
    parser.add_argument(
        "--range", type=str, default="102.4,102.4", help="detection range is [-102.4, +102.4, -102.4, +102.4]"
    )
    parser.add_argument("--no_score", action="store_true", help="whether print the score of prediction")
    parser.add_argument("--note", default="", type=str, help="any other thing?")
    parser.add_argument("--noise", type=float, default=0.0, help="add noise to pose")
    parser.add_argument("--all", action="store_true", help="evaluate all the agents instead of the first one.")
    parser.add_argument("--show_bev", action="store_true", help="Visualize the BEV feature")
    parser.add_argument(
        "--protocol_result", action="store_true", help="plot the protocol result instead of the ego result."
    )
    parser.add_argument("--data_only", action="store_true", help="Only visualize the data")
    parser.add_argument("--score_threshold", type=float, default=0.2, help="score threshold for visualization")
    parser.add_argument("--aggregation", default="", choices=["", "nms", "psa"], help="post process method")
    parser.add_argument("--task", default="detection", choices=["detection", "segmentation"], help="task type")

    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()
    hypes = yaml_utils.load_yaml(None, opt)
    hypes = update_dict(hypes, {"score_threshold": opt.score_threshold})

    if "heter" in hypes:
        x_min, x_max = -eval(opt.range.split(",")[0]), eval(opt.range.split(",")[0])
        y_min, y_max = -eval(opt.range.split(",")[1]), eval(opt.range.split(",")[1])
        opt.note += f"_{x_max}_{y_max}"
        new_cav_range = [x_min, y_min, hypes["cav_lidar_range"][2], x_max, y_max, hypes["cav_lidar_range"][5]]
        hypes = update_dict(
            hypes, {"cav_lidar_range": new_cav_range, "lidar_range": new_cav_range, "gt_range": new_cav_range}
        )   
        hypes = yaml_utils.update_yaml(hypes, opt)
    if opt.aggregation:
        hypes = update_dict(hypes, {"aggretation": opt.aggregation})

    opt.left_hand = True if ("OPV2V" in hypes["test_dir"] or "V2XSET" in hypes["test_dir"]) else False
    print(f"Left hand visualizing: {opt.left_hand}")
    if "box_align" in hypes.keys():
        hypes["box_align"]["val_result"] = hypes["box_align"]["test_result"]

    print("Creating Model")
    model = train_utils.create_model(hypes, train_flag=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading Model from checkpoint")
    saved_path = opt.model_dir
    resume_epoch, model = train_utils.load_saved_model(saved_path, model)
    print(f"resume from {resume_epoch} epoch.")
    opt.note += f"_epoch{resume_epoch}"

    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    # build dataset for each noise setting
    print("Dataset Building")
    hypes['val'] = True
    opencood_dataset = build_dataset(hypes, visualize=False, train=False)
    data_loader = DataLoader(
        opencood_dataset,
        batch_size=1,
        num_workers=8,
        collate_fn=opencood_dataset.collate_batch_train,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
    )

    criterion = train_utils.create_loss(hypes)

    opt.infer_info = opt.fusion_method + opt.note + ("_all" if opt.all else "") + "_noise" + str(opt.noise)

    loss_dict = {"m1": [], "m3": [], "m6": [], "m7": []}
    for i, batch_data in enumerate(data_loader):
        if batch_data is None:
            continue
        model.zero_grad()
        model.eval()

        batch_data = train_utils.to_device(batch_data, device)
        batch_data['ego']['epoch'] = 40
        ouput_dict = model(batch_data['ego'])

        final_loss = criterion(ouput_dict,
                                batch_data['ego']['label_dict'])
        # print(f'val loss {final_loss:.3f}')

        loss_dict[ouput_dict['backend_modality']].append(final_loss.item())
        
        if i % 100 == 0:
            for k,v in loss_dict.items():
                print(f"{k}: {np.mean(v):.3f}")
    


if __name__ == "__main__":
    main()
