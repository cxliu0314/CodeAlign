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
from torch.utils.data import DataLoader, Subset, DistributedSampler
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
from opencood.tools import multi_gpu_utils
import copy

torch.multiprocessing.set_sharing_strategy("file_system")
# CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port 6666 --use_env opencood/tools/test_val_error.py --hypes_yaml /GPFS/data/changxingliu/HEAL/opencood/hypes_yaml/opv2v/Heter_group/codebook_fusion/stage2_m1m2_m1m6.yaml --model_dir /GPFS/data/changxingliu/HEAL/opencood/logs/Codebook_Sharedhead_Pyramid_stage2_m1m2m6_2025_04_09_22_50_44

def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", "-y", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument('--fusion_method', '-f', default="intermediate",
                        help='passed to inference.')
    parser.add_argument("--half", action='store_true',
                        help="whether train with half precision")
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    opt = parser.parse_args()
    return opt

def main():
    opt = train_parser()

    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
    multi_gpu_utils.init_distributed_mode(opt)


    print("Dataset Building")    
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    hypes_val = copy.deepcopy(hypes)
    hypes_val['val'] = True
    opencood_validate_dataset = build_dataset(hypes_val, visualize=False, train=False)
    
    if opt.distributed:
        sampler_train = DistributedSampler(opencood_train_dataset)
        sampler_val = DistributedSampler(opencood_validate_dataset, shuffle=False)

        val_loader = DataLoader(opencood_validate_dataset,
                                sampler=sampler_val,
                                num_workers=8,
                                collate_fn=opencood_train_dataset.collate_batch_train,
                                drop_last=False)
    else:
        val_loader = DataLoader(opencood_validate_dataset,
                                batch_size=hypes['train_params']['batch_size'],
                                num_workers=8,
                                collate_fn=opencood_train_dataset.collate_batch_train,
                                shuffle=False,
                                pin_memory=True,
                                drop_last=True)


    print("Creating Model")
    model = train_utils.create_model(hypes, train_flag=False)
    # we assume gpu is necessary
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading Model from checkpoint")
    saved_path = opt.model_dir
    resume_epoch, model = train_utils.load_saved_model(saved_path, model)
    print(f"resume from {resume_epoch} epoch.")        

    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    # optimizer = train_utils.setup_optimizer(hypes, model_without_ddp)
    # scheduler = train_utils.setup_lr_schedular(hypes, optimizer, init_epoch=0)

    if opt.distributed:
        sampler_train.set_epoch(0)
    # model_without_ddp.model_train_init()

    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    with torch.no_grad():
        for i, batch_data in enumerate(val_loader):
            if batch_data is None:
                continue
            model.zero_grad()
            # optimizer.zero_grad()
            model.eval()
            batch_data = train_utils.to_device(batch_data, device)
            batch_data['ego']['epoch'] = 0
            ouput_dict = model(batch_data['ego'])
            final_loss = criterion(ouput_dict,
                                    batch_data['ego']['label_dict'])


if __name__ == "__main__":
    main()
