import argparse
import os
import statistics

import torch
from torch.utils.data import DataLoader, Subset

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset

from icecream import ic
import os
import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections import OrderedDict


def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", "-y", default='/GPFS/data/changxingliu/HEAL/opencood/hypes_yaml/opv2v/Heter_group/codebook_fusion/codebook_pyramid_fixenc_aligner.yaml',
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='/GPFS/data/changxingliu/HEAL/opencood/logs/preload_codebook',
                        help='Continued training path')
    parser.add_argument('--fusion_method', '-f', default="intermediate",
                        help='passed to inference.')
    opt = parser.parse_args()
    return opt

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections import OrderedDict


def visualize_combined_codebooks(pretrained_centers, clustered_centers, save_path):
    """Visualize both codebooks together using PCA"""
    # Combine all centers
    all_centers = np.vstack([pretrained_centers, clustered_centers])  # [32, 64]
    labels = ['Pretrained']*16 + ['Clustered']*16
    colors = ['red']*16 + ['blue']*16
    
    # Reduce to 2D using PCA
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(all_centers)  # [32, 2]
    
    # Create visualization
    plt.figure(figsize=(12, 10))
    
    # Plot points with different colors
    for i in range(32):
        plt.scatter(reduced[i, 0], reduced[i, 1], 
                   c=colors[i], s=10)  # Only show one label per group
    
    plt.title('Codebook Comparison (PCA-reduced)', fontsize=14)
    plt.xlabel('Principal Component 1', fontsize=12)
    plt.ylabel('Principal Component 2', fontsize=12)
    plt.legend()
    plt.grid(True)
    
    # Save figure
    vis_path = os.path.join(os.path.dirname(save_path), "codebook_comparison.png")
    plt.savefig(vis_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Codebook comparison visualization saved to {vis_path}")

def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    print('Dataset Building')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    train_loader = DataLoader(opencood_train_dataset,
                              batch_size=hypes['train_params']['batch_size'],
                              num_workers=4,
                              collate_fn=opencood_train_dataset.collate_batch_train,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True,
                              prefetch_factor=2)

    print('Creating Model')
    model = train_utils.create_model(hypes, train_flag=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if torch.cuda.is_available():
        model.to(device)

    model.train()
    feature_dict = []
    for i, batch_data in enumerate(train_loader):
        if batch_data is None or batch_data['ego']['object_bbx_mask'].sum()==0:
            continue
        if i >= 100:  # Limit to 1000 batches for speed
            break
        model.zero_grad()
        batch_data = train_utils.to_device(batch_data, device)
        heter_feature_2d = model(batch_data['ego'])

        for feature in heter_feature_2d:
            feature = feature.detach().cpu()
            feature = feature.view(64, -1)
            perm = torch.randperm(feature.size(1))[:256]
            feature_selected = feature[:, perm]  # [64, 256]
            feature_dict.append(feature_selected)
        print(f"Processed batch {i+1}/100")
    
    # Step 2: Concatenate and cluster
    all_features = torch.cat(feature_dict, dim=1).T.numpy()  # [N, 64]
    kmeans = KMeans(n_clusters=16, random_state=0).fit(all_features)
    cluster_centers = kmeans.cluster_centers_  # [16, 64]
    
    # Step 3: Prepare and save codebook
    state_dict = OrderedDict([
        ('codebook._encoders.0._quantizer._codebook', 
         torch.from_numpy(cluster_centers).float().T.unsqueeze(0))  # [64, 16]
    ])
    
    save_path = os.path.join(opt.model_dir, "preload_codebook.pth")
    torch.save(state_dict, save_path)
    print(f"Codebook saved to {save_path}")
    
    pretrained_codebook = model.state_dict()['codebook._encoders.0._quantizer._codebook']
    pretrained_centers = pretrained_codebook.squeeze(0).cpu().numpy()  # [16, 64]
    
    visualize_combined_codebooks(pretrained_centers, cluster_centers, 
                               os.path.join(opt.model_dir, "codebook_comparison.png"))

if __name__ == '__main__':
    main()
