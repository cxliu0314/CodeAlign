import argparse
import os
import statistics
import glob
import torch
from torch.utils.data import DataLoader, DistributedSampler
from tensorboardX import SummaryWriter

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import multi_gpu_utils
from icecream import ic
import tqdm
import copy
import torch.distributed as dist
import time

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 6666 --use_env opencood/tools/train_ddp.py --hypes_yaml ${CONFIG_FILE} [--model_dir  ${CHECKPOINT_FOLDER}

import random
import numpy as np

def set_random_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}. This ensures reproducibility across runs.")

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
    import sys
    class Tee(object):
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()

    set_random_seeds(42)
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
    multi_gpu_utils.init_distributed_mode(opt)

    if opt.model_dir:
        saved_path = opt.model_dir
    else:
        init_epoch = 0
        if dist.get_rank() == 0:
            saved_path = train_utils.setup_train(hypes)
        else:
            saved_path = None
        saved_path_list = [saved_path]
        dist.broadcast_object_list(saved_path_list, src=0)
        saved_path = saved_path_list[0]
        
    log_path = os.path.join(saved_path, 'log.log')
    log_file = open(log_path, 'a', buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)

    print('Dataset Building')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    hypes_val = copy.deepcopy(hypes)
    hypes_val['val'] = True
    opencood_validate_dataset = build_dataset(hypes_val,
                                              visualize=False,
                                              train=False)

    if opt.distributed:
        sampler_train = DistributedSampler(opencood_train_dataset)
        sampler_val = DistributedSampler(opencood_validate_dataset, shuffle=False)

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, hypes['train_params']['batch_size'], drop_last=True)

        train_loader = DataLoader(opencood_train_dataset,
                                  batch_sampler=batch_sampler_train,
                                  num_workers=8,
                                  collate_fn=opencood_train_dataset.collate_batch_train)
        val_loader = DataLoader(opencood_validate_dataset,
                                sampler=sampler_val,
                                num_workers=8,
                                collate_fn=opencood_train_dataset.collate_batch_train,
                                drop_last=False)
    else:
        train_loader = DataLoader(opencood_train_dataset,
                                  batch_size=hypes['train_params'][
                                      'batch_size'],
                                  num_workers=8,
                                  collate_fn=opencood_train_dataset.collate_batch_train,
                                  shuffle=True,
                                  pin_memory=True,
                                  drop_last=True)
        val_loader = DataLoader(opencood_validate_dataset,
                                batch_size=hypes['train_params']['batch_size'],
                                num_workers=8,
                                collate_fn=opencood_train_dataset.collate_batch_train,
                                shuffle=True,
                                pin_memory=True,
                                drop_last=True)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # record lowest validation loss checkpoint.
    lowest_val_loss = 1e5
    lowest_val_epoch = -1

    # if we want to train from last checkpoint.
    if opt.model_dir:
        init_epoch, model = train_utils.load_saved_model(saved_path, model)
        lowest_val_epoch = init_epoch
    else:
        init_epoch = 0

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)
        
    # ddp setting
    model_without_ddp = model

    if opt.distributed:
        model = \
            torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[opt.gpu],
                                                      find_unused_parameters=True) # True
        model_without_ddp = model.module


    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    # optimizer = train_utils.setup_optimizer(hypes, model_without_ddp)
    optimizer = train_utils.setup_optimizer_codebook(hypes, model_without_ddp)
    
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer, init_epoch=init_epoch)

    # record training
    writer = SummaryWriter(saved_path)

    # half precision training
    if opt.half:
        scaler = torch.cuda.amp.GradScaler()

    print('Training start')
    epoches = hypes['train_params']['epoches']
    supervise_single_flag = False if not hasattr(opencood_train_dataset, "supervise_single") else opencood_train_dataset.supervise_single
    # used to help schedule learning rate

    # add data balance for code2code module
    data_balance = [10 for _ in hypes['model']['args']['backend_modality']]
    for epoch in range(init_epoch, max(epoches, init_epoch)):
        # cal val when load model
        valid_ave_loss = []
        loss_list = [[] for _ in hypes['model']['args']['backend_modality']]
        if opt.model_dir and epoch==init_epoch:
            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    if batch_data is None:
                        continue
                    model.zero_grad()
                    optimizer.zero_grad()
                    model.eval()
                    batch_data = train_utils.to_device(batch_data, device)
                    batch_data['ego']['epoch'] = epoch
                    ouput_dict = model(batch_data['ego'])
                    final_loss = criterion(ouput_dict, batch_data['ego']['label_dict'], suffix='val')
                    valid_ave_loss.append(final_loss.item())
                    backend_modality = ouput_dict.get('backend_modality', None)
                    if backend_modality is not None:
                        index = hypes['model']['args']['backend_modality'].index(backend_modality)
                        loss_list[index].append(final_loss.item())
                    else:
                        print('Wrong backend modality during validation.')
                    criterion.logging(epoch, i, len(val_loader), None, suffix="val")
            lowest_val_loss = statistics.mean(valid_ave_loss)
            print('At epoch %d, the validation loss is %f' % (epoch, lowest_val_loss))
            print('At epoch %d, the loss list is %s' % (epoch, loss_list))
            print('At epoch %d, the data balance is %s' % (epoch, data_balance))

        # start training
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])
        if opt.distributed:
            sampler_train.set_epoch(epoch)
        # the model will be evaluation mode during validation
        model.train()
        try: # heter_model stage2
            model_without_ddp.model_train_init()
        except:
            print("No model_train_init function")
        for i, batch_data in enumerate(train_loader):
            if batch_data is None or batch_data['ego']['object_bbx_mask'].sum()==0:
                continue
            model.zero_grad()
            optimizer.zero_grad()
            batch_data = train_utils.to_device(batch_data, device)
            batch_data['ego']['epoch'] = epoch
            if not opt.half:
                ouput_dict = model(batch_data['ego'], data_balance=data_balance)
                final_loss = criterion(ouput_dict,
                                       batch_data['ego']['label_dict'])
            else:
                with torch.cuda.amp.autocast():
                    ouput_dict = model(batch_data['ego'], data_balance=data_balance)
                    final_loss = criterion(ouput_dict, batch_data['ego']['label_dict'])

            criterion.logging(epoch, i, len(train_loader), writer)

            if supervise_single_flag:
                if not opt.half:
                    final_loss += criterion(ouput_dict, batch_data['ego']['label_dict_single'], suffix="_single") * hypes['train_params'].get("single_weight", 1)
                else:
                    with torch.cuda.amp.autocast():
                        final_loss += criterion(ouput_dict, batch_data['ego']['label_dict_single'], suffix="_single") * hypes['train_params'].get("single_weight", 1)
                criterion.logging(epoch, i, len(train_loader), writer, suffix="_single")

            if not opt.half:
                final_loss.backward()
                optimizer.step()
            else:
                scaler.scale(final_loss).backward()
                scaler.step(optimizer)
                scaler.update()

        # torch.cuda.empty_cache() # it will destroy memory buffer
        if (epoch+1) % hypes['train_params']['save_freq'] == 0:
            torch.save(model_without_ddp.state_dict(),
                       os.path.join(saved_path,
                                    'net_epoch%d.pth' % (epoch + 1)))
            
        if (epoch+1) % hypes['train_params']['eval_freq'] == 0:
            valid_ave_loss = []

            from collections import defaultdict
            modality_losses = defaultdict(list)
            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    if batch_data is None:
                        continue
                    model.zero_grad()
                    optimizer.zero_grad()
                    model.eval()

                    batch_data = train_utils.to_device(batch_data, device)
                    batch_data['ego']['epoch'] = epoch
                    ouput_dict = model(batch_data['ego'])

                    final_loss = criterion(ouput_dict,
                                           batch_data['ego']['label_dict'], suffix='val')
                    valid_ave_loss.append(final_loss.item())
                    backend_modality = ouput_dict.get('backend_modality', None)
                    if backend_modality is not None:
                        if backend_modality in hypes['model']['args']['backend_modality']:
                            modality_losses[backend_modality].append(final_loss.item())
                        else:
                            print(f'Unknown backend modality: {backend_modality}')
                    else:
                        print('Wrong backend modality during validation.')
                    criterion.logging(epoch, i, len(val_loader), None, suffix="val")

            valid_ave_loss = statistics.mean(valid_ave_loss)
            print('At epoch %d, the validation loss is %f' % (epoch,
                                                              valid_ave_loss))
            writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)

            # === 数据平衡调整 ===
            loss_list = []  # 存储每个模态的平均 loss
            for mod in hypes['model']['args']['backend_modality']:
                if mod in modality_losses and len(modality_losses[mod]) > 0:
                    avg_loss = np.mean(modality_losses[mod])
                else:
                    avg_loss = 0.0  # 或者设为全局平均、或跳过，这里设为 0 可能影响平衡，也可设为 valid_ave_loss
                    print(f"Warning: No data for modality {mod}, using 0.0 for balance.")
                loss_list.append(avg_loss)
            # 计算平均 loss（避免空列表）
            if len(loss_list) > 0:
                avg_l = np.mean(loss_list)
            else:
                avg_l = 0.0
            # 调整 data_balance
            for d_i in range(len(data_balance)):
                if loss_list[d_i] < avg_l:
                    data_balance[d_i] -= 1
                elif loss_list[d_i] > avg_l:
                    data_balance[d_i] += 1
                else:
                    pass
                if data_balance[d_i] < 1:
                    data_balance[d_i] = 1
            print('At epoch %d, the loss list is %s' % (epoch, loss_list))
            print('At epoch %d, the data balance is %s' % (epoch, data_balance))

            # lowest val loss
            if valid_ave_loss < lowest_val_loss:
                lowest_val_loss = valid_ave_loss
                torch.save(model_without_ddp.state_dict(),
                       os.path.join(saved_path,
                                    'net_epoch_bestval_at%d.pth' % (epoch + 1)))
                if lowest_val_epoch != -1 and os.path.exists(os.path.join(saved_path,
                                    'net_epoch_bestval_at%d.pth' % (lowest_val_epoch))):
                    if opt.rank == 0:
                        os.remove(os.path.join(saved_path,
                                        'net_epoch_bestval_at%d.pth' % (lowest_val_epoch)))
                lowest_val_epoch = epoch + 1

        scheduler.step(epoch)
        
        opencood_train_dataset.reinitialize()

    print('Training Finished, checkpoints saved to %s' % saved_path)
    dist.barrier()  # 其他进程等待主进程保存完成

    if opt.rank == 0:
        run_test = True
        
        # ddp training may leave multiple bestval
        bestval_model_list = glob.glob(os.path.join(saved_path, "net_epoch_bestval_at*"))
        
        if len(bestval_model_list) > 1:
            bestval_model_epoch_list = [eval(x.split("/")[-1].lstrip("net_epoch_bestval_at").rstrip(".pth")) for x in bestval_model_list]
            ascending_idx = np.argsort(bestval_model_epoch_list)
            for idx in ascending_idx:
                if idx != (len(bestval_model_list) - 1):
                    os.remove(bestval_model_list[idx])

        if run_test:
            fusion_method = opt.fusion_method
            cmd = f"python opencood/tools/inference_heter_task_average.py --model_dir {saved_path}"
            print(f"Running command: {cmd}")
            os.system(cmd)


if __name__ == '__main__':
    main()
