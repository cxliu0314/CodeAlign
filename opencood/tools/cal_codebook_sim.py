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
from opencood.utils.common_utils import update_dict
from matplotlib import pyplot as plt
from tqdm import tqdm
import cv2
import numpy as np
from opencood.utils.common_utils import read_json

def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--model_dir", type=str, default="", help="Continued training path")
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

def get_feature(model_dir, modal):
    opt = test_parser()
    opt.model_dir = model_dir
    hypes = yaml_utils.load_yaml(None, opt)
    hypes = update_dict(hypes, {"score_threshold": opt.score_threshold})

    if "heter" in hypes:
        x_min, x_max = -eval(opt.range.split(",")[0]), eval(opt.range.split(",")[0])
        y_min, y_max = -eval(opt.range.split(",")[1]), eval(opt.range.split(",")[1])
        opt.note += f"_{x_max}_{y_max}"
        new_cav_range = [x_min, y_min, hypes["cav_lidar_range"][2], x_max, y_max, hypes["cav_lidar_range"][5]]
        hypes = update_dict(hypes, {"cav_lidar_range": new_cav_range, "lidar_range": new_cav_range, "gt_range": new_cav_range})
        hypes = yaml_utils.update_yaml(hypes, opt)

    hypes["validate_dir"] = hypes["test_dir"]
    if "OPV2V" in hypes["test_dir"] or "v2xsim" in hypes["test_dir"]: assert "test" in hypes["validate_dir"]
    opt.left_hand = True if ("OPV2V" in hypes["test_dir"] or "V2XSET" in hypes["test_dir"]) else False
    if "box_align" in hypes.keys(): hypes["box_align"]["val_result"] = hypes["box_align"]["test_result"]

    print("Creating Model")
    model = train_utils.create_model(hypes, train_flag=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved_path = opt.model_dir
    resume_epoch, model = train_utils.load_saved_model(saved_path, model)
    print(f"resume from {resume_epoch} epoch.")
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    modal_assign = read_json(hypes['heter']['assignment_path'])
    for scene in modal_assign.keys():
        for u, cav_id in enumerate(modal_assign[scene]):
            modal_assign[scene][cav_id] = modal
    hypes['modality_assignment'] = modal_assign

    print("Dataset Building")
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    data_loader = DataLoader(
        opencood_dataset,
        batch_size=1,
        num_workers=8,
        collate_fn=opencood_dataset.collate_batch_test,
        shuffle=False
    )

    for i, batch_data in enumerate(data_loader):
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            cav_content = batch_data["ego"]
            vis_dict = model(cav_content, vis_path=True)
            break
    return vis_dict, model, batch_data["ego"]['label_dict']['foreground_map']

def main():
    # m2
    model_dir1 = '/GPFS/data/changxingliu/HEAL/opencood/logs/homo_codebook/m2_16'
    vis_dict1, model1, fg_mask = get_feature(model_dir1, 'm2')
    
    # m1
    model_dir2 = '/GPFS/data/changxingliu/HEAL/opencood/logs/homo_codebook/m1_16'
    vis_dict2, model2, fg_mask = get_feature(model_dir2, 'm1')
    
    codemap1 = vis_dict1['code_warp'][0][0].cpu().detach().numpy()
    codemap2 = vis_dict2['code_warp'][0][0].cpu().detach().numpy()

    count_matrix = np.zeros((16, 16), dtype=int)
    for i in range(256):
        for j in range(256):
            if fg_mask[0][i,j]!=0:
                count_matrix[int(codemap1[i, j]), int(codemap2[i, j])] += 1
    plt.figure(figsize=(8, 8))
    cax = plt.imshow(count_matrix, cmap='Blues', interpolation='nearest')
    plt.colorbar(cax)
    for i in range(16):
        for j in range(16):
            plt.text(j, i, f"{count_matrix[i, j]}", ha='center', va='center', color='black', fontsize=6)
    plt.title("Correspondence of Codemap1 and Codemap2")
    plt.xlabel("m1 Index")
    plt.ylabel("m2 Index")
    image_path = "/GPFS/data/changxingliu/HEAL/opencood/logs/feature_vis_0618/sim_space_fg.png"
    plt.savefig(image_path)
    plt.close()

    import ipdb; ipdb.set_trace()

    # cosine similarity
    # tensor = torch.eye(16).unsqueeze(1).to('cuda')
    # feature1 = model1.multi_channel_compressor.get_feature([tensor]).cpu() # torch.Size([n, 1, 16])
    # feature2 = model2.multi_channel_compressor.get_feature([tensor]).cpu() # torch.Size([n, 1, 16])
    # feature1 = model1.multi_channel_compressor.Codebooks[0][0].cpu()
    # feature2 = model2.multi_channel_compressor.Codebooks[0][0].cpu()

    # feature1_norm = feature1 / feature1.norm(dim=1, keepdim=True)
    # feature2_norm = feature2 / feature2.norm(dim=1, keepdim=True)
    # similarity_matrix = torch.mm(feature1_norm, feature2_norm.T)

    # plt.figure(figsize=(6, 6))
    # plt.imshow(similarity_matrix.detach().numpy(), cmap='gray')
    # for i in range(similarity_matrix.size(0)):
    #     for j in range(similarity_matrix.size(1)):
    #         plt.text(j, i, f"{similarity_matrix[i, j]:.2f}", ha='center', va='center', color='white', fontsize=8)
    # plt.title("Similarity Matrix")
    # plt.xlabel("m1 Index")
    # plt.ylabel("m2 Index")
    # image_path = "/GPFS/data/changxingliu/HEAL/opencood/logs/feature_vis_0618/sim_codebook.png"
    # plt.savefig(image_path)
    # plt.close()

    
    
    
if __name__ == "__main__":
    main()
