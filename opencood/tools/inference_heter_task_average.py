# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>, Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>,
# License: TDG-Attribution-NonCommercial-NoDistrib

# Modifications by Xiangbo Gao <xiangbogaobarry@gmail.com>
# New License for modifications: MIT License

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

    # if opt.protocol_result:
    #     # No need to plot BEV feature when plotting protocol result, the BEV feature is plotted in the ego mode.
    #     opt.show_bev = False
    #     return opt

    return opt

# CUDA_VISIBLE_DEVICES=3 python opencood/tools/inference_heter_task_average.py --model_dir 

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

        # reload ego modality for diff backend
        backend_modality = hypes['model']['args'].get("backend_modality", None)
        ego_modality_ori = hypes['heter']['ego_modality']
        if backend_modality is not None:
            hypes['heter']['ego_modality'] = list(set(backend_modality) | set(ego_modality_ori))

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

    if opt.noise:
        # add noise to pose.
        pos_std = opt.noise
        rot_std = opt.noise
        pos_mean = 0
        rot_mean = 0

        # setting noise
        np.random.seed(303)
        noise_setting = OrderedDict()
        noise_args = {"pos_std": pos_std, "rot_std": rot_std, "pos_mean": pos_mean, "rot_mean": rot_mean}

        noise_setting["add_noise"] = True
        noise_setting["args"] = noise_args

        # build dataset for each noise setting
        print("Dataset Building")
        print(f"Noise Added: {pos_std}/{rot_std}/{pos_mean}/{rot_mean}.")
        hypes.update({"noise_setting": noise_setting})

    import itertools
    import copy
    from opencood.utils.common_utils import read_json
    
    heter_group = hypes['heter']['heter_group']
    backend_modality = hypes['model']['args'].get("backend_modality", None)
    if backend_modality is not None:
        heter_infer_pair = [[b, e] for b in backend_modality for e in ego_modality_ori]
    elif len(heter_group)==1:
        if len(heter_group[0])==1:
            heter_infer_pair = [[heter_group[0][0], heter_group[0][0]]]
        else: 
            heter_infer_pair = list(itertools.permutations(heter_group[0], 2))
            heter_infer_pair = [list(pair) for pair in heter_infer_pair]
    else:
        # 去除不同组之间的重复元素
        heter_infer_group = []
        for i in range(len(heter_group)):
            for j in range(i + 1, len(heter_group)):
                group1, group2 = set(heter_group[i]), set(heter_group[j])
                group1_unique = list(group1 - group2)
                group2_unique = list(group2 - group1)
                heter_infer_group.append((group1_unique, group2_unique))
        # 使用 itertools.product 进行两两组的组合
        heter_infer_pair = []
        for group1, group2 in heter_infer_group:
            heter_infer_pair.extend([list(pair) for pair in itertools.product(group1, group2)])
        # 添加反向组合
        heter_infer_pair = heter_infer_pair + [list(pair[::-1]) for pair in heter_infer_pair]
    print('heter_infer_pair', heter_infer_pair)
    # heter_infer_pair=[['m6', 'm2']]


    result_list = []
    for pair in heter_infer_pair:
        hypes_iter = copy.deepcopy(hypes)
        modal_assign = read_json(hypes_iter['heter']['assignment_path'])
        for scene in modal_assign.keys():
            for u, cav_id in enumerate(modal_assign[scene]):
                if u==0:
                    modal_assign[scene][cav_id]=pair[0]
                else:
                    modal_assign[scene][cav_id]=pair[1]
        hypes_iter['modality_assignment'] = modal_assign

        # build dataset for each noise setting
        print("Dataset Building")
        opencood_dataset = build_dataset(hypes_iter, visualize=True, train=False)
        data_loader = DataLoader(
            opencood_dataset,
            batch_size=1,
            num_workers=8,
            collate_fn=opencood_dataset.collate_batch_test,
            shuffle=False,
            pin_memory=False,
            drop_last=False,
        )

        modality_list = opencood_dataset.modality_name_list
        result_stat = {
            0.3: {"tp": [], "fp": [], "gt": 0, "score": []},
            0.5: {"tp": [], "fp": [], "gt": 0, "score": []},
            0.7: {"tp": [], "fp": [], "gt": 0, "score": []},
        }

        opt.infer_info = opt.fusion_method + opt.note + ("_all" if opt.all else "") + "_noise" + str(opt.noise) + '_' + pair[0] + '_' + pair[1]

        pbar = tqdm(enumerate(data_loader))
        for i, batch_data in pbar:
            pbar.set_description(f"{opt.infer_info}_{i}")
            if batch_data is None:
                continue

            if opt.data_only:
                os.makedirs(os.path.join(opt.model_dir, "data"), exist_ok=True)
                simple_vis_stamp.visualize(
                    None,
                    batch_data["ego"]["origin_lidar"][0],
                    new_cav_range,
                    os.path.join(opt.model_dir, "data", f"lidar_{i}.png"),
                    method="bev",
                    left_hand=opt.left_hand,
                )
                continue

            with torch.no_grad():
                batch_data = train_utils.to_device(batch_data, device)

                if opt.fusion_method == "late":
                    infer_result = inference_utils_stamp.inference_late_fusion(batch_data, model, opencood_dataset)
                elif opt.fusion_method == "early":
                    infer_result = inference_utils_stamp.inference_early_fusion(batch_data, model, opencood_dataset)
                elif opt.fusion_method == "intermediate":
                    infer_result = inference_utils_stamp.inference_intermediate_fusion(
                        batch_data,
                        model,
                        opencood_dataset,
                        infer_all=opt.all,
                        show_bev=opt.show_bev,
                        protocol_result=opt.protocol_result,
                    )
                elif opt.fusion_method == "no":
                    infer_result = inference_utils_stamp.inference_no_fusion(batch_data, model, opencood_dataset)
                elif opt.fusion_method == "no_w_uncertainty":
                    infer_result = inference_utils_stamp.inference_no_fusion_w_uncertainty(batch_data, model, opencood_dataset)
                elif opt.fusion_method == "single":
                    infer_result = inference_utils_stamp.inference_no_fusion(batch_data, model, opencood_dataset, single_gt=True)
                elif opt.fusion_method == "late_heter":
                    infer_result = inference_utils_stamp.inference_heter_late(
                        batch_data,
                        model,
                        opencood_dataset,
                        show_bev=opt.show_bev,
                        infer_all=opt.all,
                    )
                else:
                    raise NotImplementedError(
                        "Only single, no, no_w_uncertainty, early, late and intermediate" "fusion is supported."
                    )

                agent_modality_list = batch_data["ego"]["agent_modality_list"]
                if not opt.all:
                    infer_result = [infer_result]

                for idx, infer_result_single in enumerate(infer_result):
                    work_dir = opt.model_dir
                    eval_detection_result(
                        opt,
                        agent_modality_list,
                        opencood_dataset,
                        infer_result_single,
                        result_stat,
                        batch_data,
                        idx,
                        work_dir,
                        hypes_iter,
                        i,
                        pair,
                        model
                    )

            torch.cuda.empty_cache()
        
        ap30, ap50, ap70 = eval_utils.eval_final_results(result_stat, opt.model_dir, opt.infer_info)
        result_list.append([ap30, ap50, ap70])

    result_array = np.array(result_list)
    ap_30 = np.mean(result_array[:, 0])  # 平均 ap30
    ap_50 = np.mean(result_array[:, 1])  # 平均 ap50
    ap_70 = np.mean(result_array[:, 2])  # 平均 ap70
    print('The Average Precision at IOU 0.3 is %.4f, '
          'The Average Precision at IOU 0.5 is %.4f, '
          'The Average Precision at IOU 0.7 is %.4f' % (ap_30, ap_50, ap_70))
    
    eval_path = os.path.join(opt.model_dir, 'eval_average.json')
    eval_data = {
        "Average Precision at IOU 0.3": round(ap_30, 6),
        "Average Precision at IOU 0.5": round(ap_50, 6),
        "Average Precision at IOU 0.7": round(ap_70, 6)
    }
    with open(eval_path, 'w') as f:
        json.dump(eval_data, f, indent=4)


def eval_detection_result(
    opt, agent_modality_list, opencood_dataset, infer_result_single, result_stat, batch_data, idx, work_dir, hypes, i, pair, model
):

    pred_box_tensor = infer_result_single["pred_box_tensor"]
    gt_box_tensor = infer_result_single["gt_box_tensor"]
    pred_score = infer_result_single["pred_score"]
    if pred_box_tensor is None or gt_box_tensor is None or pred_score is None:
        return
    eval_utils.caluclate_tp_fp(
        pred_box_tensor,
        pred_score,
        gt_box_tensor,
        result_stat[agent_modality_list[idx]] if opt.all else result_stat,
        0.3,
    )
    eval_utils.caluclate_tp_fp(
        pred_box_tensor,
        pred_score,
        gt_box_tensor,
        result_stat[agent_modality_list[idx]] if opt.all else result_stat,
        0.5,
    )
    eval_utils.caluclate_tp_fp(
        pred_box_tensor,
        pred_score,
        gt_box_tensor,
        result_stat[agent_modality_list[idx]] if opt.all else result_stat,
        0.7,
    )
    if opt.save_npy:
        npy_save_path = os.path.join(work_dir, "npy")
        if not os.path.exists(npy_save_path):
            os.makedirs(npy_save_path)
        inference_utils_stamp.save_prediction_gt(
            pred_box_tensor, gt_box_tensor, batch_data["ego"]["origin_lidar"][0], i, npy_save_path
        )

    if not opt.no_score:
        infer_result_single.update({"score_tensor": pred_score})

    if getattr(opencood_dataset, "heterogeneous", False):
        cav_box_np, agent_modality_list = inference_utils_stamp.get_cav_box(batch_data)
        infer_result_single.update({"cav_box_np": cav_box_np, "agent_modality_list": agent_modality_list})

    if (i % opt.save_vis_interval == 0) and (pred_box_tensor is not None or gt_box_tensor is not None):
        vis_save_path_root = os.path.join(work_dir, f'vis_{opt.infer_info}_{pair[0]}_{pair[1]}')
        if not os.path.exists(vis_save_path_root):
            os.makedirs(vis_save_path_root)
        vis_save_path = os.path.join(vis_save_path_root, "bev_%05d.png" % i)
        try:
            # new version considering various gt ranges
            gt_range = hypes["heter"]["modality_setting"][infer_result_single["ego_modality"]]["postprocess"]["gt_range"]
        except:
            gt_range = hypes["postprocess"]["gt_range"]
        try:
            pcd_modality = batch_data["ego"]["origin_lidar_modality"][0]
        except:
            pcd_modality = torch.tensor(1).cuda()
        simple_vis_stamp.visualize(
            infer_result_single,
            batch_data["ego"]["origin_lidar"][0],
            gt_range,
            vis_save_path,
            method="bev",
            transformation_matrix_clean=torch.inverse(batch_data["ego"]["transformation_matrix_clean"].cpu()).cuda(),
            transformation_matrix=torch.inverse(batch_data["ego"]["transformation_matrix"].cpu()).cuda(),
            left_hand=opt.left_hand,
            show_bev=opt.show_bev,
            pcd_modality=pcd_modality,
        )

        # vis_feature(batch_data, model, vis_save_path_root, i)

            # transformation_matrix_clean=torch.inverse(batch_data["ego"]["transformation_matrix_clean"][idx].cpu()).cuda(),
            # transformation_matrix=torch.inverse(batch_data["ego"]["transformation_matrix"][idx].cpu()).cuda(),
        

def vis_feature(batch_data, model, save_path, i):
    cav_content = batch_data["ego"]
    vis_dict = model(cav_content, vis_path=save_path)
    for key, tensor in vis_dict.items():
        if 'encoder' in key or 'aligner' in key:
            N, C, H, W = tensor.shape
            mean_tensor = tensor.mean(dim=1)
            fig, axes = plt.subplots(1, N, figsize=(5 * N, 5), squeeze=False)
            axes = axes[0]
            for n_i in range(N):
                img = mean_tensor[n_i].cpu().numpy()  # [H, W]
                axes[n_i].imshow(img, cmap='viridis')
                axes[n_i].axis('off')
                axes[n_i].set_title(f'car {n_i}')
            plt.tight_layout()
            plt.savefig(f'{save_path}/bev_%05d_{key}.png' % i, bbox_inches='tight', pad_inches=0.1, dpi=300)
            plt.close()
            continue

        if key not in ['aligner', 'codebook', 'codebook_crop', 'codebook_warp', 'fusion']: continue

        N, C, H, W = tensor.shape
        mean_tensor = tensor.mean(dim=1)
        fig, axes = plt.subplots(1, N, figsize=(5 * N, 5), squeeze=False)
        axes = axes[0]
        for n_i in range(N):
            img = mean_tensor[n_i].cpu().numpy()  # [H, W]
            axes[n_i].imshow(img, cmap='viridis')
            axes[n_i].axis('off')
            axes[n_i].set_title(f'car {n_i}')
        plt.tight_layout()
        plt.savefig(f'{save_path}/bev_%05d_{key}.png' % i, bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close()


if __name__ == "__main__":
    main()
