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

    for epoch in range(init_epoch, max(epoches, init_epoch)):
        # cal val when load model
        if opt.model_dir and epoch==init_epoch:
            valid_ave_loss = []
            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    if batch_data is None: continue
                    model.zero_grad()
                    optimizer.zero_grad()
                    model.eval()
                    batch_data = train_utils.to_device(batch_data, device)
                    batch_data['ego']['epoch'] = epoch
                    ouput_dict = model(batch_data['ego'])
                    final_loss = criterion(ouput_dict, batch_data['ego']['label_dict'], suffix='val')
                    valid_ave_loss.append(final_loss.item())
                    criterion.logging(epoch, i, len(val_loader), None, suffix="val")
            lowest_val_loss = statistics.mean(valid_ave_loss)
            print('At epoch %d, the validation loss is %f' % (epoch, lowest_val_loss))

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
                ouput_dict = model(batch_data['ego'])
                final_loss = criterion(ouput_dict,
                                       batch_data['ego']['label_dict'])
            else:
                with torch.cuda.amp.autocast():
                    ouput_dict = model(batch_data['ego'])
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
                    criterion.logging(epoch, i, len(val_loader), None, suffix="val")

            valid_ave_loss = statistics.mean(valid_ave_loss)
            print('At epoch %d, the validation loss is %f' % (epoch,
                                                              valid_ave_loss))
            writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)

            # lowest val loss
            if valid_ave_loss < lowest_val_loss:
                lowest_val_loss = valid_ave_loss
                if opt.rank == 0:
                    torch.save(model_without_ddp.state_dict(),
                        os.path.join(saved_path, 'net_epoch_bestval_at%d.pth' % (epoch + 1)))
                # if lowest_val_epoch != -1 and os.path.exists(os.path.join(saved_path,
                #                     'net_epoch_bestval_at%d.pth' % (lowest_val_epoch))):
                #     if opt.rank == 0:
                #         os.remove(os.path.join(saved_path,
                #                         'net_epoch_bestval_at%d.pth' % (lowest_val_epoch)))
                lowest_val_epoch = epoch + 1

        scheduler.step(epoch)
        
        opencood_train_dataset.reinitialize()

    print('Training Finished, checkpoints saved to %s' % saved_path)
    # dist.barrier()  # 其他进程等待主进程保存完成

    if opt.rank == 0:
        run_test = True
        
        # ddp training may leave multiple bestval
        bestval_model_list = glob.glob(os.path.join(saved_path, "net_epoch_bestval_at*.pth"))
    
        if len(bestval_model_list) > 1:
            try:
                # 创建epoch到文件路径的映射
                epoch_to_file = {}
                
                # 遍历所有文件，安全地解析epoch号
                for file_path in bestval_model_list:
                    try:
                        filename = os.path.basename(file_path)
                        # 使用字符串方法安全地提取epoch号
                        epoch_str = filename.replace("net_epoch_bestval_at", "").replace(".pth", "")
                        epoch_num = int(epoch_str)
                        epoch_to_file[epoch_num] = file_path
                    except (ValueError, AttributeError) as e:
                        print(f"Warning: Cannot parse epoch from filename {file_path}: {e}")
                        continue
                
                # 如果成功解析了至少两个文件的epoch号，则进行清理
                if len(epoch_to_file) > 1:
                    # 获取所有epoch号并排序
                    sorted_epochs = sorted(epoch_to_file.keys())
                    # 保留最大的epoch对应的文件，删除其他文件
                    latest_epoch = sorted_epochs[-1]
                    for epoch in sorted_epochs[:-1]:
                        try:
                            file_to_remove = epoch_to_file[epoch]
                            if os.path.exists(file_to_remove):
                                os.remove(file_to_remove)
                                print(f"Removed old bestval model: {file_to_remove}")
                        except Exception as e:
                            print(f"Warning: Failed to remove {file_to_remove}: {e}")
                    print(f"保留最新的bestval模型: {epoch_to_file[latest_epoch]}")
            except Exception as e:
                print(f"Error during bestval model cleanup: {e}")

        # 运行推理
        if run_test:
            fusion_method = opt.fusion_method
            cmd = f"python opencood/tools/inference_heter_task_average.py --model_dir {saved_path}"
            print(f"Running command: {cmd}")
            os.system(cmd)


if __name__ == '__main__':
    main()