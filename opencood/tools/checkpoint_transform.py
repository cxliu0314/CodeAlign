import torch

def convert_checkpoint(model1_path, model2_path, output_path, name_mapping):
    """
    转换模型参数命名空间并保存新checkpoint

    参数:
        model1_path: str, model1的checkpoint路径（目标结构）
        model2_path: str, model2的checkpoint路径（源参数）
        output_path: str, 输出checkpoint路径
        name_mapping: dict, 前缀映射字典 {model1前缀: model2前缀}
    """
    # 加载两个checkpoint
    model1_ckpt = torch.load(model1_path, map_location='cpu')
    model2_ckpt = torch.load(model2_path, map_location='cpu')

    # 获取state_dict
    model1_state_dict = model1_ckpt['model'] if 'model' in model1_ckpt else model1_ckpt
    model2_state_dict = model2_ckpt['model'] if 'model' in model2_ckpt else model2_ckpt

    new_state_dict = {}
    missing_keys = []
    redundant_keys = []

    # 转换参数命名空间
    for key1 in model1_state_dict.keys():
        # 分割第一段前缀和剩余部分
        parts = key1.split('.', 1)
        prefix = parts[0]
        suffix = parts[1] if len(parts) > 1 else ""

        # 应用名称映射
        mapped_prefix = name_mapping.get(prefix, prefix)
        key2 = f"{mapped_prefix}.{suffix}" if suffix else mapped_prefix

        if key2 in model2_state_dict:
            new_state_dict[key1] = model2_state_dict[key2]
        else:
            new_state_dict[key1] = model1_state_dict[key1]  # 保留原值
            missing_keys.append((key1, key2))

    # 检查model2中未使用的参数
    used_keys = {f"{name_mapping.get(k.split('.')[0], k.split('.')[0])}.{'.'.join(k.split('.')[1:])}" 
                for k in model1_state_dict.keys()}
    redundant_keys = [k for k in model2_state_dict.keys() if k not in used_keys]

    # 更新checkpoint
    if 'model' in model1_ckpt:
        model1_ckpt['model'] = new_state_dict
    else:
        model1_ckpt = new_state_dict

    # 保存转换后的checkpoint
    torch.save(model1_ckpt, output_path)

    # 打印转换报告
    print(f"Checkpoint 转换完成，已保存到: {output_path}")
    print(f"总参数数量: {len(model1_state_dict)}")
    
    if missing_keys:
        print("\n警告: 以下参数在model2中未找到对应项:")
        for key1, key2 in missing_keys:
            print(f"  - Model1 参数名: {key1}")
            print(f"    映射到 Model2 参数名: {key2} (未找到)")
    
    if redundant_keys:
        print("\n注意: model2中存在以下未使用的参数:")
        for key in redundant_keys:  # 只显示前10个避免过多输出
            print(f"  - {key}")


model1_path = '/GPFS/data/changxingliu/HEAL/opencood/logs/C2C_Classifier_ConvNeXT3_Codebook16_m2tom1_2025_06_18_15_10_09/net_epoch_bestval_at4.pth'
model2_path = '/GPFS/data/changxingliu/HEAL/opencood/logs/homo_codebook/m1m2_16/net_epoch_bestval_at9.pth'
new_path = '/GPFS/data/changxingliu/HEAL/opencood/logs/homo_codebook/m1m2_16/converted_ckpt.pth'

print("Model1 keys:", [k.split('.')[0] for k in torch.load(model1_path).keys()])
print("Model2 keys:", [k.split('.')[0] for k in torch.load(model2_path).keys()])

name_mapping = {
    "encoder_m2": "encoder_m2",
    "backbone_m2": "backbone_m2",
    "translator": "code2code_m2",
    "codebook": "multi_channel_compressor_m0",
    "fusion_backbone": "pyramid_backbone_m0",
    "shrink_conv": "shrink_conv_m0",
    "cls_head": "cls_head_m0",
    "reg_head": "reg_head_m0",
    "dir_head": "dir_head_m0",
}

convert_checkpoint(model1_path, model2_path, new_path, name_mapping)