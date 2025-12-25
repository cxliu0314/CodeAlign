import torch
import torch.nn as nn
import torch.optim as optim
import glob
import importlib
import yaml
import re
import argparse
import os
import statistics
import random
import numpy as np
from torch.utils.data import DataLoader
from opencood.models.heter_pointpillars_lift_splat_v2_with_feature import HeterPointPillarsLiftSplatV2withfeature
import opencood.hypes_yaml.yaml_utils as yaml_utils


def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", "-y", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='opencood/logs/get_feature/',
                        help='Continued training path')
    parser.add_argument('--fusion_method', '-f', default="intermediate",
                        help='passed to inference.')
    parser.add_argument('--lidar_ratio', default=0.5,
                        help='lidar ratio when using heter dataset')      
    parser.add_argument('--ego_modality', default='random',
                        help='choose the ego modality of dataset')      
    parser.add_argument('--seg_num', default=2,
                        help='the seg_num of codebook')                        
    parser.add_argument('--dict_size', default=128,
                        help='the real dict_size of codebook will be [dict_size, dict_size, dict_size]')            
    opt = parser.parse_args()
    return opt
    

class featDataset(torch.utils.data.Dataset):
    def __init__(self, path='/GPFS/rhome/sifeiliu/OpenCOODv2/opencood/logs/feature_folder/'):
        file_names = os.listdir(path)
        self.data = []
        num = 0
        for file_name in file_names:
            num = num + 1
            if 'feature' in file_name and num % 100 == 1:
                print(file_name)
                self.data.append(torch.load(path+file_name))

    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)


opt = train_parser()
hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Load Model")
model = HeterPointPillarsLiftSplatV2withfeature(hypes['model']['args'])
# pretrained_path = 'opencood/logs/cb_only/v2v4real/0719_model_14.pth'
# model.load_state_dict(torch.load(pretrained_path, map_location='cpu'), strict=False)
model = model.to(device)
learning_rate = 0.0001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 40

model_save_path = opt.model_dir
print("Training Start")
for epoch in range(num_epochs):
    print("Load Data")
    featdataset = featDataset()
    featdataloader = DataLoader(featdataset, batch_size=1, shuffle=False, num_workers=8)
    if epoch == 40 or epoch == 80 or epoch == 100:
        learning_rate = learning_rate / 10
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print("Begin Training")
    for i, data in enumerate(featdataloader):
        model.train()
        optimizer.zero_grad()  
        data = data.to(device)[0]
        out_dict = model(data)
        print('epoch: {}, iter: {}, loss: {}'.format(epoch, i, out_dict['codebook_loss']))
        loss = out_dict['codebook_loss']

        loss.backward()  
        optimizer.step()  

    torch.save(model.state_dict(), model_save_path+'0719_model_lowerLR_{}.pth'.format(epoch))