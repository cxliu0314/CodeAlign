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

        # reload ego modality for diff backend
        backend_modality = hypes['model']['args'].get("backend_modality", None)
        ego_modality_ori = hypes['heter']['ego_modality']
        if backend_modality is not None:
            hypes['heter']['ego_modality'] = list(set(backend_modality) | set(ego_modality_ori))

        hypes = yaml_utils.update_yaml(hypes, opt)
    if opt.aggregation:
        hypes = update_dict(hypes, {"aggretation": opt.aggregation})

    hypes["validate_dir"] = hypes["test_dir"]
    if "OPV2V" in hypes["test_dir"] or "v2xsim" in hypes["test_dir"]:
        assert "test" in hypes["validate_dir"]
    opt.left_hand = True if ("OPV2V" in hypes["test_dir"] or "V2XSET" in hypes["test_dir"]) else False
    print(f"Left hand visualizing: {opt.left_hand}")
    if "box_align" in hypes.keys():
        hypes["box_align"]["val_result"] = hypes["box_align"]["test_result"]

    print("Creating Model")
    model = train_utils.create_model(hypes, train_flag=False)
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
            if u==0:
                modal_assign[scene][cav_id] = 'm1'
            else:
                modal_assign[scene][cav_id] = 'm2'
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

    from matplotlib.colors import ListedColormap
    def visualize_code_maps(code_maps, foreground_mask, save_path, filename="combined_codemap.png"):
        N, C, H, W = code_maps.shape
        code_cmap = plt.cm.get_cmap('tab20', 16)  # 16 colors for codes
        mask_cmap = ListedColormap(['black', 'red'])  # Binary: 0=black, 1=red
        
        fig, axes = plt.subplots(1, N + 1, figsize=(5*(N + 1), 5), squeeze=False)
        axes = axes[0]  # Flatten axes array
        for n in range(N):
            code_map = code_maps[n].squeeze().cpu().numpy()  # (H, W)
            im = axes[n].imshow(code_map, cmap=code_cmap, vmin=0, vmax=15)
            axes[n].set_title(f'Code {n}')
            axes[n].axis('off')
        
        mask = foreground_mask[0]  # Use first sample's mask if batch > 1
        axes[-1].imshow(mask, cmap=mask_cmap, vmin=0, vmax=1)
        axes[-1].set_title('Foreground Mask')
        axes[-1].axis('off')
        
        fig.colorbar(im, ax=axes[:-1], orientation='horizontal', 
                    ticks=np.arange(0, 16), label='Code Index',
                    pad=0.05, fraction=0.05)
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/{filename}", bbox_inches='tight', dpi=300)
        plt.close()


    save_path = os.path.join(opt.model_dir, 'feature_vis')
    print(save_path)
    os.makedirs(save_path, exist_ok=True)
    n=0
    pbar = tqdm(enumerate(data_loader))
    for i, batch_data in pbar:
        pbar.set_description(f"{opt.infer_info}_{i}")
        if batch_data is None:
            continue
        if i%40 != 0:
            continue

        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            cav_content = batch_data["ego"]
            if len(cav_content['agent_modality_list']) > 1:
                vis_dict = model(cav_content, vis_path=save_path)

                foreground_map = batch_data["ego"]['label_dict']['foreground_map']
                positives = batch_data["ego"]['label_dict']['pos_equal_one']  # [B, H, W, 2]
                foreground_mask = torch.logical_or(positives[..., 0], positives[..., 1]).float()  # [B, H, W]
        
                visualize_code_maps(vis_dict['codemap_warp'], foreground_map, save_path, filename=f"{i}_codemap.png")
                
                for key, tensor in vis_dict.items():
                    if 'encoder' in key:
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
                        plt.savefig(f'{save_path}/{i}_{key}.png', 
                                    bbox_inches='tight', pad_inches=0.1, dpi=300)
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
                    plt.savefig(f'{save_path}/{i}_{key}.png', 
                                bbox_inches='tight', pad_inches=0.1, dpi=300)
                    plt.close()

                print(f"Save feature to {save_path}")
                n+=1
            else:
                print(cav_content['agent_modality_list'])

        if n > 10:
            break


if __name__ == "__main__":
    main()
