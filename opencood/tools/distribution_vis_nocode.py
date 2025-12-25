# -*- coding: utf-8 -*-
import argparse
import os
import json
import statistics
import time
from typing import OrderedDict
import importlib
import torch
import torchvision
import open3d as o3d
from torch.utils.data import DataLoader, Subset
import numpy as np
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils_stamp
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.visualization import vis_utils, my_vis, simple_vis_stamp
from opencood.utils.common_utils import update_dict
from opencood.utils.seg_iou import mean_IU
from matplotlib import pyplot as plt
from tqdm import tqdm
import cv2
import numpy as np

torch.multiprocessing.set_sharing_strategy("file_system")
# CUDA_VISIBLE_DEVICES=7 python opencood/tools/distribution_vis.py --model_dir opencood/logs/test

def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--model_dir", type=str, required=True, help="Continued training path")
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

def umap(features, n_neighbors=30, min_dist=0.1):
    import umap.umap_ as umap
    features_np = features.cpu().numpy() if isinstance(features, torch.Tensor) else features
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42) #, unique=True
    embedding = reducer.fit_transform(features_np)
    return embedding

def pca(features):
    from sklearn.decomposition import PCA
    features_np = features.cpu().numpy() if isinstance(features, torch.Tensor) else features
    reducer = PCA(n_components=2)
    embedding = reducer.fit_transform(features_np)
    return embedding

def tsne(features, perplexity=30):
    from sklearn.manifold import TSNE
    features_np = features.cpu().numpy() if isinstance(features, torch.Tensor) else features
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=500,
        random_state=42,
        init='pca'
    )
    embedding = tsne.fit_transform(features_np)
    return embedding

from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple
from opencood.models.fuse_modules.fusion_in_one import regroup
def warp_feature(feature, affine_matrix, record_len, mode='bilinear'):
    N, C, H, W = feature.shape
    B, L = affine_matrix.shape[:2]

    split_x = regroup(feature, record_len)
    feature_transed = []
    for b in range(B):
        t_matrix = affine_matrix[b][:N, :N, :, :]
        feature_in_ego = warp_affine_simple(split_x[b], t_matrix[0, :, :, :], (H, W), align_corners=False, mode=mode)
        feature_transed.append(feature_in_ego)
    feature = torch.cat(feature_transed, dim=0) 
    
    return feature

def main():
    opt = test_parser()

    assert opt.fusion_method in ["late", "late_heter", "early", "intermediate", "no", "no_w_uncertainty", "single"]
    # if opt.all:
    #     assert not opt.show_bev

    hypes = yaml_utils.load_yaml(None, opt)

    hypes = update_dict(
        hypes,
        {
            "score_threshold": opt.score_threshold,
        },
    )

    if "heter" in hypes:
        # hypes['heter']['lidar_channels'] = 16
        # opt.note += "_16ch"

        x_min, x_max = -eval(opt.range.split(",")[0]), eval(opt.range.split(",")[0])
        y_min, y_max = -eval(opt.range.split(",")[1]), eval(opt.range.split(",")[1])
        opt.note += f"_{x_max}_{y_max}"

        new_cav_range = [x_min, y_min, hypes["cav_lidar_range"][2], x_max, y_max, hypes["cav_lidar_range"][5]]
        # replace all appearance
        hypes = update_dict(
            hypes, {"cav_lidar_range": new_cav_range, "lidar_range": new_cav_range, "gt_range": new_cav_range}
        )

        # reload anchor
        hypes = yaml_utils.update_yaml(hypes, opt)

    if opt.aggregation:
        hypes = update_dict(hypes, {"aggretation": opt.aggregation})

    hypes["validate_dir"] = hypes["test_dir"]
    if "OPV2V" in hypes["test_dir"] or "v2xsim" in hypes["test_dir"]:
        assert "test" in hypes["validate_dir"]

    # This is used in visualization
    # left hand: OPV2V, V2XSet
    # right hand: V2X-Sim 2.0 and DAIR-V2X
    opt.left_hand = True if ("OPV2V" in hypes["test_dir"] or "V2XSET" in hypes["test_dir"]) else False

    print(f"Left hand visualizing: {opt.left_hand}")

    if "box_align" in hypes.keys():
        hypes["box_align"]["val_result"] = hypes["box_align"]["test_result"]

    print("Creating Model")
    model = train_utils.create_model(hypes, train_flag=False)
    # we assume gpu is necessary
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading Model from checkpoint")
    saved_path = opt.model_dir
    resume_epoch, model = train_utils.load_saved_model(saved_path, model)
    print(f"resume from {resume_epoch} epoch.")
    opt.note += f"_epoch{resume_epoch}"

    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    
    from opencood.utils.common_utils import read_json

    modal_assign = read_json(hypes['heter']['assignment_path'])
    for scene in modal_assign.keys():
        for u, cav_id in enumerate(modal_assign[scene]):
            # if u==0:
            modal_assign[scene][cav_id] = 'm1'
            # elif u==1:
            #     modal_assign[scene][cav_id] = 'm2'
            # elif u==2:
            #     modal_assign[scene][cav_id] = 'm6'
            # else:
            #     modal_assign[scene][cav_id] = 'm1'
    hypes['modality_assignment'] = modal_assign

    # build dataset for each noise setting
    print("Dataset Building")
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    data_loader = DataLoader(
        opencood_dataset,
        batch_size=1,
        num_workers=8,
        collate_fn=opencood_dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )

    opt.infer_info = opt.fusion_method + opt.note + ("_all" if opt.all else "") + "_noise" + str(opt.noise)

    pbar = tqdm(enumerate(data_loader))
    save_path = opt.model_dir+'/dist_vis'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    n=0
    for i, batch_data in pbar:
        pbar.set_description(f"{opt.infer_info}_{i}")
        if batch_data is None:
            continue
        if i % 50 != 0: continue
        # if i<170: continue

        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            cav_content = batch_data["ego"]
            if True:#cav_content['agent_modality_list']==['m1', 'm2', 'm6']:
                ouput_dict = model(cav_content)
                affine_matrix = ouput_dict['affine_matrix']
                record_len = batch_data['ego']['record_len']
                B, L = affine_matrix.shape[:2]

                feature_before = ouput_dict['feature_before'] # N, C, H, W
                N, C, H, W = feature_before.shape
                fused_feature = ouput_dict['fused_feature'] # B, C, H, W

                positives = batch_data['ego']['label_dict']['pos_equal_one']
                foreground_mask = torch.logical_or(positives[..., 0], positives[..., 1]).float() # [b,h,w]

                # warp to ego coordinate
                feature_before = warp_feature(feature_before, affine_matrix, record_len) # N, C, H, W

                feature_before = feature_before.permute(0, 2, 3, 1).view(N, H*W, C).cpu().numpy()
                fused_feature = fused_feature.permute(0, 2, 3, 1).view(B, H*W, C).cpu().numpy()
                masks = foreground_mask.view(B, -1).cpu().numpy()
                masks = masks[0]

                for method in ['pca', 'umap_15', 'umap_30', 'umap_100', 'tsne_5', 'tsne_30', 'tsne_100']:
                    if method == 'pca':
                        feature_embedding = pca(feature_before.reshape(-1, C))
                        fused_embedding = pca(fused_feature.reshape(-1, C))
                    elif method == 'umap_15':
                        feature_embedding = umap(feature_before.reshape(-1, C), n_neighbors=15, min_dist=0.1)
                        fused_embedding = umap(fused_feature.reshape(-1, C), n_neighbors=15, min_dist=0.1)
                    elif method == 'umap_30':
                        feature_embedding = umap(feature_before.reshape(-1, C), n_neighbors=30, min_dist=0.1)
                        fused_embedding = umap(fused_feature.reshape(-1, C), n_neighbors=30, min_dist=0.1)
                    elif method == 'umap_100':
                        feature_embedding = umap(feature_before.reshape(-1, C), n_neighbors=100, min_dist=0.1)
                        fused_embedding = umap(fused_feature.reshape(-1, C), n_neighbors=100, min_dist=0.1)
                    elif method == 'tsne_5':
                        feature_embedding = tsne(feature_before.reshape(-1, C), perplexity=5)
                        fused_embedding = tsne(fused_feature.reshape(-1, C), perplexity=5)
                    elif method == 'tsne_30':
                        feature_embedding = tsne(feature_before.reshape(-1, C), perplexity=30)
                        fused_embedding = tsne(fused_feature.reshape(-1, C), perplexity=30)
                    elif method == 'tsne_100':
                        feature_embedding = tsne(feature_before.reshape(-1, C), perplexity=100)
                        fused_embedding = tsne(fused_feature.reshape(-1, C), perplexity=100)

                    feature_embedding = feature_embedding.reshape(N, H*W, 2)

                    plt.figure(figsize=(6*(N+1), 6))
                    for n in range(N+1):
                        if n==N:
                            emb = fused_embedding
                        else:
                            emb = feature_embedding[n]

                        plt.subplot(1, N+1, n+1)
                        plt.scatter(emb[masks==0, 0], emb[masks==0, 1], s=5, c='gray', alpha=0.5, label='Background')
                        plt.scatter(emb[masks==1, 0], emb[masks==1, 1], s=20, c='red', alpha=0.8, marker='X', label='Foreground')
                        if n==N:
                            plt.title(f'Fused Feature - {method}', fontsize=20)
                        else:
                            plt.title(f'Car {n+1} - {cav_content["agent_modality_list"][n]} - Encoded Feature - {method}', fontsize=20)
                        plt.grid(alpha=0.2)
                        plt.legend(loc='upper right', fontsize=10)
                    plt.tight_layout()
                    plt.savefig(save_path+f'/distribution_{i}_{method}.jpg', dpi=600, bbox_inches='tight')
                    plt.close()
                    print(f"Saved to {save_path+f'/distribution_{i}_{method}.jpg'}")
                        
                    # for n in range(N):
                    #     before_idx = n
                    #     after_idx = N + n

                    #     # Before特征子图
                    #     plt.subplot(N, 3, 3*n + 1)
                    #     emb_before = combined_embedding[before_idx]
                    #     plt.scatter(emb_before[masks[0]==0, 0], emb_before[masks[0]==0, 1], s=5, c='gray', alpha=0.5, label='Background')
                    #     plt.scatter(emb_before[masks[0]==1, 0], emb_before[masks[0]==1, 1], s=20, c='red', alpha=0.8, marker='X', label='Foreground')
                    #     plt.title(f'Car {n+1} - {cav_content["agent_modality_list"][n]} - Before Codebook', fontsize=20)
                    #     plt.grid(alpha=0.2)
                    #     plt.legend(loc='upper right', fontsize=10)

                    #     # After特征子图
                    #     plt.subplot(N, 3, 3*n + 2)
                    #     emb_after = combined_embedding[after_idx]
                    #     plt.scatter(emb_after[masks[0]==0, 0], emb_after[masks[0]==0, 1], s=5, c='gray', alpha=0.5, label='Background')
                    #     plt.scatter(emb_after[masks[0]==1, 0], emb_after[masks[0]==1, 1], s=20, c='red', alpha=0.8, marker='X', label='Foreground')
                    #     plt.title(f'Car {n+1} - {cav_content["agent_modality_list"][n]} - After Codebook', fontsize=20)
                    #     plt.grid(alpha=0.2)
                    #     plt.legend(loc='upper right', fontsize=10)

                    #     # Codebook子图
                    #     plt.subplot(N, 3, 3*n + 3)
                    #     plt.scatter(code_embedding[index_mask==0, 0], code_embedding[index_mask==0, 1], s=10, c='gray', alpha=0.5, label='Background')
                    #     plt.scatter(code_embedding[index_mask==0.5, 0], code_embedding[index_mask==0.5, 1], s=20, c='blue', alpha=0.8, marker='X', label='B & F')
                    #     plt.scatter(code_embedding[index_mask==1, 0], code_embedding[index_mask==1, 1], s=30, c='red', alpha=0.8, marker='X', label='Foreground')
                    #     plt.title(f'Car {n+1} - {cav_content["agent_modality_list"][n]} - Codebook', fontsize=20)
                    #     plt.grid(alpha=0.2)
                    #     plt.legend(loc='upper right', fontsize=10)

                    # plt.tight_layout()
                    # plt.savefig(save_path+f'/combined_distribution_{method}_{i}.jpg', dpi=600, bbox_inches='tight')
                    # plt.close()
                    # print(f"Saved to {save_path+f'/combined_distribution_{method}_{i}.jpg'}")


                n+=1

        if n > 10:
            break


if __name__ == "__main__":
    main()
