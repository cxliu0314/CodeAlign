# -*- coding: utf-8 -*-
# Author: Xiangbo Gao <xiangbogaobarry@gmail.com>
# License: MIT License
"""
这段代码是用于 训练或加载模型并保存其状态字典 的一个 Python 脚本。
它利用了 argparse 解析命令行参数，从指定路径加载模型，并根据指定的配置文件和适配器路径更新模型，最后将模型的状态字典保存到文件中。

1. 加载模型配置：根据提供的路径加载配置文件和模型。
2. 加载适配器和代理模型：遍历适配器目录和代理路径，加载并更新模型。
3. 加载协议模型（如果指定）：如果启用了 --with_protocol，加载协议模型。
4. 保存模型状态字典：将模型的状态字典保存到指定路径，以供后续使用。
"""

import argparse
import os
from opencood.tools import train_utils
import opencood.hypes_yaml.yaml_utils as yaml_utils
import torch
import re
import glob
from opencood.tools.train_utils import check_missing_key

# 需要使用 完整路径
# ln -s /GPFS/rhome/zichenchao/czc_python_works/STAMP/opencood/logs/opv2v/object_detection/final_infer/protocol_m1/net_epoch1.pth /GPFS/rhome/zichenchao/czc_python_works/STAMP/opencood/logs/opv2v/object_detection/final_infer/protocol_m1/m6m1m6m6
rename_names_list = ['reg_head', 'shrink_conv', 'fusion_backbone', 'cls_head', 'dir_head']
m1_ckpt_path = 'opencood/logs/homo/m1/net_epoch_bestval_at40.pth'
m2_ckpt_path = 'opencood/logs/homo/m2/net_epoch_bestval_at42.pth'
m3_ckpt_path = 'opencood/logs/homo/m3/net_epoch_bestval_at34.pth'
m4_ckpt_path = 'opencood/logs/homo/m4/net_epoch_bestval_at32.pth'
m5_ckpt_path = 'opencood/logs/homo/m5/net_epoch_bestval_at18.pth'
m6_ckpt_path = 'opencood/logs/homo/m6/net_epoch_bestval_at18.pth'
m7_ckpt_path = 'opencood/logs/homo/m7/net_epoch_bestval_at18.pth'

save_path = 'opencood/logs_czc/opv2v/latefusion'

# ========== 关键修改：合并四个模型的state_dict（按新逻辑处理） ==========
# 初始化合并后的状态字典
merged_state_dict = {}

# 定义处理单个模型的函数（避免重复代码）
def process_model(ckpt_path, suffix):
    loaded_state_dict = torch.load(ckpt_path, map_location='cpu')
    # 遍历当前模型的所有键值对
    for k, v in loaded_state_dict.items():
        new_k = k  # 默认保留原key
        # 检查是否包含需要重命名的关键词
        for rename_key in rename_names_list:
            if rename_key in k:
                # 替换关键词并更新key
                new_k = k.replace(rename_key, rename_key + suffix)
                break  # 避免同一key被多个关键词重复替换
        # 将处理后的键值对加入合并字典
        merged_state_dict[new_k] = v

# 依次处理四个模型
process_model(m1_ckpt_path, '_m1')
process_model(m2_ckpt_path, '_m2')
process_model(m3_ckpt_path, '_m3')
process_model(m4_ckpt_path, '_m4')
process_model(m5_ckpt_path, '_m5')
process_model(m6_ckpt_path, '_m6')
process_model(m7_ckpt_path, '_m7')

# ========== 保存合并后的状态字典 ==========
os.makedirs(save_path, exist_ok=True)
torch.save(merged_state_dict, os.path.join(save_path, 'net_epoch1.pth'))

# 可选：打印合并信息，验证结果
print(f"合并完成！总参数个数：{len(merged_state_dict)}")
for suffix in ['_m1', '_m2','_m3', '_m4','_m5', '_m6', '_m7']:
    count = len([k for k in merged_state_dict.keys() if suffix in k])
    print(f"{suffix[1:]}模型参数个数（含重命名和未重命名）：{count}")

