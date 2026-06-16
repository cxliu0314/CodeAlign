"""
-*- coding: utf-8 -*-
Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
License: TDG-Attribution-NonCommercial-NoDistrib

Incrementally increase heterogeneous agents in order.

Actual collaborator m1 -> m1+m2 -> m1+m2+m3 -> m1+m2+m3+m4

Ego is always m1

commrange is 180 (large enough)

For Intermediate Fusion, we will switch to IntermediateHeterinferFusionDataset
"""

import argparse
import os
import time
from typing import OrderedDict
import importlib
import torch
import open3d as o3d
from torch.utils.data import DataLoader, Subset
import numpy as np
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.visualization import vis_utils, my_vis, simple_vis, simple_vis_stamp
from opencood.utils.common_utils import update_dict
torch.multiprocessing.set_sharing_strategy('file_system')



def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', type=str,
                        default='intermediate',
                        help='no, no_w_uncertainty, late, early or intermediate')
    parser.add_argument('--save_vis_interval', type=int, default=40,
                        help='interval of saving visualization')
    parser.add_argument('--no_vis', action='store_true',
                        help='disable visualization output; useful for read-only checkpoint dirs')
    parser.add_argument('--output_dir', type=str, default='',
                        help='directory for eval/visualization outputs; defaults to model_dir')
    parser.add_argument('--legacy_model_init', action='store_true',
                        help='use train_utils.create_model(hypes), matching the historical czc inference entry')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    parser.add_argument('--range', type=str, default="204.8,102.4",
                        help="detection range is [-204.8, +204.8, -102.4, +102.4]")
    parser.add_argument('--no_score', action='store_true',
                        help="whether print the score of prediction")
    parser.add_argument('--use_cav', type=str, default="[1,2,3,4]",
                        help="evaluate with real collaborator number")
    parser.add_argument('--lidar_degrade', action='store_true',
                        help="whether to degrade lidar. {m1:32, m3:16} and {m1:16, m3:16}")
    parser.add_argument('--note', default="", type=str, help="any other thing?")
    parser.add_argument("--noise", type=float, default=0.0, help="add noise to pose")
    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()
    output_dir = opt.output_dir or opt.model_dir
    os.makedirs(output_dir, exist_ok=True)

    assert opt.fusion_method in ['late', 'early', 'intermediate', 'no', 'no_w_uncertainty', 'single'] 

    hypes = yaml_utils.load_yaml(None, opt)

    if 'heter' in hypes:
        # hypes['heter']['lidar_channels'] = 16
        # opt.note += "_16ch"

        x_min, x_max = -eval(opt.range.split(',')[0]), eval(opt.range.split(',')[0])
        y_min, y_max = -eval(opt.range.split(',')[1]), eval(opt.range.split(',')[1])
        opt.note += f"_{x_max}_{y_max}"

        new_cav_range = [x_min, y_min, hypes['postprocess']['anchor_args']['cav_lidar_range'][2], \
                            x_max, y_max, hypes['postprocess']['anchor_args']['cav_lidar_range'][5]]

        # replace all appearance
        hypes = update_dict(hypes, {
            "cav_lidar_range": new_cav_range,
            "lidar_range": new_cav_range,
            "gt_range": new_cav_range
        })

        # hypes = update_dict(hypes, {
        #     "mapping_dict": {
        #         "m1": "m1",
        #         "m2": "m2",
        #         "m3": "m3",
        #         "m4": "m4"
        #     }
        # })
        
        # reload ego modality for diff backend
        backend_modality = hypes['model']['args'].get("backend_modality", None)
        ego_modality_ori = hypes['heter']['ego_modality']
        if backend_modality is not None:
            hypes['heter']['ego_modality'] = list(set(backend_modality) | set(ego_modality_ori))

        # reload anchor
        yaml_utils_lib = importlib.import_module("opencood.hypes_yaml.yaml_utils")
        for name, func in yaml_utils_lib.__dict__.items():
            if name == hypes["yaml_parser"]:
                parser_func = func
        hypes = parser_func(hypes)

        
    
    hypes['validate_dir'] = hypes['test_dir']
    if "OPV2V" in hypes['test_dir'] or "v2xsim" in hypes['test_dir']:
        assert "test" in hypes['validate_dir']
    
    # This is used in visualization
    # left hand: OPV2V, V2XSet
    # right hand: V2X-Sim 2.0 and DAIR-V2X
    left_hand = True if ("OPV2V" in hypes['test_dir'] or "V2XSET" in hypes['test_dir']) else False

    print(f"Left hand visualizing: {left_hand}")

    if 'box_align' in hypes.keys():
        hypes['box_align']['val_result'] = hypes['box_align']['test_result']

    print('Creating Model')
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
    if opt.legacy_model_init:
        model = train_utils.create_model(hypes)
    else:
        model = train_utils.create_model(hypes, train_flag=False)
    # we assume gpu is necessary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    resume_epoch, model = train_utils.load_saved_model(saved_path, model)
    print(f"resume from {resume_epoch} epoch.")
    opt.note += f"_epoch{resume_epoch}"
    
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    # setting noise
    np.random.seed(303)

    if opt.fusion_method == 'intermediate':
        hypes['fusion']['core_method'] += 'infer' 
    hypes['comm_range'] = 180
    if hypes['heter'].get('assignment_path',None) is not None:
        hypes['heter']['assignment_path'] = hypes['heter']['assignment_path'].replace(".json", "_in_order.json")
    # hypes = update_dict(hypes, {
    #         "ego_modality": 'm1'
    #     })
    
    if opt.lidar_degrade:
        lidar_dict1 = {
            "m1": 32,
            "m3": 16
        }
        lidar_dict2 = {
            "m1": 16,
            "m3": 16
        }
        opt.use_cav = "[4]"
        use_cav_and_lidar_config_pair = [(4, lidar_dict1), (4, lidar_dict2)]
    else:
        lidar_dict0 = {
            'm3': 32
        }
        use_cav_and_lidar_config_pair = [(x, lidar_dict0) for x in eval(opt.use_cav)]

    for (use_cav, lidar_config) in use_cav_and_lidar_config_pair:
        hypes['use_cav'] = use_cav
        if lidar_config is not None:
            hypes['heter']['lidar_channels_dict'] = lidar_config
            print(hypes['heter']['lidar_channels_dict'])

        # build dataset for each noise setting
        print('Dataset Building')
        opencood_dataset = build_dataset(hypes, visualize=True, train=False)
        # opencood_dataset_subset = Subset(opencood_dataset, range(1220,1260))
        # data_loader = DataLoader(opencood_dataset_subset,
        data_loader = DataLoader(opencood_dataset,
                                batch_size=1,
                                num_workers=4,
                                collate_fn=opencood_dataset.collate_batch_test,
                                shuffle=False,
                                pin_memory=False,
                                drop_last=False)
        
        # Create the dictionary for evaluation
        result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                    0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                    0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}

        
        infer_info = opt.fusion_method + opt.note + f"_use_cav{use_cav}"
        if opt.lidar_degrade:
            infer_info += f"_m1_{lidar_config['m1']}_m3_{lidar_config['m3']}"
        if opt.noise != 0:
            infer_info = f'pos_{opt.noise}_rot_{opt.noise}' + opt.note + f"_use_cav{use_cav}"


        for i, batch_data in enumerate(data_loader):
            # we can just save the batch_data to file. For perfomance test.
            # intermediate dataset
            # command: CUDA_VISIBLE_DEVICES=1 python opencood/tools/inference_heter_in_order.py --model_dir opencood/logs/opv2v/Heal/noise_infer/m1m7m2 --range 102.4,102.4 --use_cav [3] --noise 0.2
            # import pickle
            # print(batch_data['ego']['agent_modality_list'])
            # collabortor = "".join(batch_data['ego']['agent_modality_list'])
            # save_dir = f"opencood/logs/FLOPs_calc/{collabortor}_online"
            # with open(os.path.join(save_dir, 'input.pkl'), 'wb') as f:
            #     pickle.dump(batch_data, f)
            # break

            # late fusion dataset
            # command:  
            # python opencood/tools/inference_heter_in_order.py --model_dir opencood/logs/opv2v/latefusion/m1m2m7_noise --fusion_method late --use_cav [1] --range 102.4,102.4 --noise 1.0
            # python opencood/tools/inference_heter_task_average.py --model_dir opencood/logs/opv2v/Heal/final_infer/m1m7m2m6 --range 102.4,102.4
            # import pickle
            # for i_, (cav_id, cav_content) in enumerate(batch_data.items()):
            #     modality_name = cav_content['modality_name']
            #     print(cav_id, modality_name)
            #     save_dir = f"opencood/logs/FLOPs_calc/{modality_name}_single"
            #     with open(os.path.join(save_dir, 'input.pkl'), 'wb') as f:
            #         pickle.dump({'ego':cav_content}, f)
            #     if i_ >= 4:
            #         break
            # raise

            print(f"{infer_info}_{i}")
            if batch_data is None:
                continue
            with torch.no_grad():
                batch_data = train_utils.to_device(batch_data, device)

                if opt.fusion_method == 'late':
                    infer_result = inference_late_fusion_heter_in_order(batch_data,
                                                            model,
                                                            opencood_dataset,
                                                            use_cav)
                elif opt.fusion_method == 'intermediate':
                    infer_result = inference_utils.inference_intermediate_fusion(batch_data,
                                                                    model,
                                                                    opencood_dataset)
                elif opt.fusion_method == 'no':
                    infer_result = inference_utils.inference_no_fusion(batch_data,
                                                                    model,
                                                                    opencood_dataset)
                elif opt.fusion_method == 'single':
                    infer_result = inference_utils.inference_no_fusion(batch_data,
                                                                    model,
                                                                    opencood_dataset,
                                                                    single_gt=True)
                else:
                    raise NotImplementedError('Only single, no, no_w_uncertainty, early, late and intermediate'
                                            'fusion is supported.')

                pred_box_tensor = infer_result['pred_box_tensor']
                gt_box_tensor = infer_result['gt_box_tensor']
                pred_score = infer_result['pred_score']
                
                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                        pred_score,
                                        gt_box_tensor,
                                        result_stat,
                                        0.3)
                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                        pred_score,
                                        gt_box_tensor,
                                        result_stat,
                                        0.5)
                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                        pred_score,
                                        gt_box_tensor,
                                        result_stat,
                                        0.7)

                if not opt.no_score:
                    infer_result.update({'score_tensor': pred_score})

                if getattr(opencood_dataset, "heterogeneous", False):
                    cav_box_np, agent_modality_list = inference_utils.get_cav_box(batch_data)
                    infer_result.update({"cav_box_np": cav_box_np, \
                                        "agent_modality_list": agent_modality_list})

                if (not opt.no_vis) and (i % opt.save_vis_interval == 0) and (pred_box_tensor is not None or gt_box_tensor is not None):
                    vis_save_path_root = os.path.join(output_dir, f'vis_{infer_info}')
                    if not os.path.exists(vis_save_path_root):
                        os.makedirs(vis_save_path_root)

                    vis_save_path = os.path.join(vis_save_path_root, 'bev_%05d.png' % i)
                    
                    try:
                        # new version considering various gt ranges
                        gt_range = hypes["heter"]["modality_setting"][infer_result["ego_modality"]]["postprocess"]["gt_range"]
                    except:
                        gt_range = hypes["postprocess"]["gt_range"]
                    try:
                        pcd_modality = batch_data["ego"]["origin_lidar_modality"][0]
                    except:
                        pcd_modality = torch.tensor(1).cuda()
                    simple_vis_stamp.visualize(
                        infer_result,
                        batch_data["ego"]["origin_lidar"][0],
                        gt_range,
                        vis_save_path,
                        method="bev",
                        transformation_matrix_clean=torch.inverse(batch_data["ego"]["transformation_matrix_clean"].cpu()).cuda(),
                        transformation_matrix=torch.inverse(batch_data["ego"]["transformation_matrix"].cpu()).cuda(),
                        left_hand=left_hand,
                        show_bev=False,
                        pcd_modality=pcd_modality,
                    )

                    # Feature visualization is only supported by CodeAlign
                    # translator models, not by legacy late-fusion baselines.
                    # vis_feature(batch_data, model, vis_save_path_root, i)
                    # simple_vis.visualize(infer_result,
                    #                     batch_data['ego'][
                    #                         'origin_lidar'][0],
                    #                     hypes['postprocess']['gt_range'],
                    #                     vis_save_path,
                    #                     method='bev',
                    #                     left_hand=left_hand)
            torch.cuda.empty_cache()

        _, ap50, ap70 = eval_utils.eval_final_results(result_stat,
                                    output_dir, infer_info)




def inference_late_fusion_heter_in_order(batch_data, model, dataset, use_cav):
    """
    Model inference for late fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.LateFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()

    # ['ego', "650", "659", ...]  keys in batch_data is in order
    for i_, (cav_id, cav_content) in enumerate(batch_data.items()):
        if i_ >= use_cav:
            break
        output_dict[cav_id] = model(cav_content)

    pred_box_tensor, pred_score, gt_box_tensor = \
        dataset.post_process(batch_data,
                             output_dict)

    return_dict = {"pred_box_tensor" : pred_box_tensor, \
                    "pred_score" : pred_score, \
                    "gt_box_tensor" : gt_box_tensor}
    return return_dict


from matplotlib.colors import ListedColormap
from matplotlib import pyplot as plt
def visualize_code_maps(code_maps, foreground_mask, save_path):
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
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def vis_feature(batch_data, model, save_path, i):
    cav_content = batch_data["ego"]
    vis_dict = model(cav_content, vis_path=save_path)
    foreground_map = batch_data["ego"]['label_dict']['foreground_map']
    positives = batch_data["ego"]['label_dict']['pos_equal_one']  # [B, H, W, 2]

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
    visualize_code_maps(vis_dict['codemap_warp'], foreground_map, f'{save_path}/bev_%05d_codemap.png' % i)


if __name__ == '__main__':
    main()
