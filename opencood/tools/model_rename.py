import os
import torch
import shutil
from collections import OrderedDict

def rename_model_arbitrary_prefix(model_path, prefix_mapping):
    """
    加载模型，按任意前缀映射重命名 state_dict 的键，并安全保存。

    Args:
        model_path (str): 模型文件路径（.pth）
        prefix_mapping (dict): {old_prefix: new_prefix}
            例如: {'backend_models.m1': 'm1', 'backbone.stem': 'stem'}
            支持完整 key 匹配或带子模块的前缀匹配。
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    print(f"🔍 正在加载模型: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')

    # 提取 state_dict 并判断是否包裹
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        has_wrapper = True
        other_fields = {k: v for k, v in checkpoint.items() if k != 'state_dict'}
    else:
        # 假设整个 checkpoint 就是 state_dict（或可转为 dict）
        if not isinstance(checkpoint, dict):
            if hasattr(checkpoint, 'state_dict'):
                state_dict = checkpoint.state_dict()
            else:
                raise ValueError("无法提取 state_dict：模型格式不支持")
        else:
            state_dict = checkpoint
        has_wrapper = False
        other_fields = {}

    # 打印第一层级模块（用于参考）
    first_level_modules = sorted(set(key.split('.')[0] for key in state_dict.keys()))
    print("\n📂 模型第一层级模块（供参考）:")
    for name in first_level_modules:
        print(f"  - {name}")

    # 按前缀长度降序排序，确保长前缀优先匹配（避免 'a' 匹配到 'a.b'）
    sorted_mappings = sorted(prefix_mapping.items(), key=lambda x: len(x[0]), reverse=True)

    # 构建新 state_dict
    new_state_dict = OrderedDict()
    renamed_count = 0

    for key, value in state_dict.items():
        new_key = None
        for old_prefix, new_prefix in sorted_mappings:
            if key == old_prefix:
                new_key = new_prefix
                break
            elif key.startswith(old_prefix + '.'):
                # 替换前缀，保留后缀
                new_key = new_prefix + key[len(old_prefix):]
                break
        if new_key is not None:
            new_state_dict[new_key] = value
            renamed_count += 1
        else:
            new_state_dict[key] = value

    print(f"\n✅ 共重命名 {renamed_count} 个参数")

    # 构建新 checkpoint
    if has_wrapper:
        new_checkpoint = {'state_dict': new_state_dict}
        new_checkpoint.update(other_fields)  # 保留 epoch, optimizer 等
    else:
        new_checkpoint = new_state_dict

    # 安全保存：先备份原文件
    # 修复路径构建逻辑 - 只在文件名前添加前缀，保留原目录结构
    dir_path = os.path.dirname(model_path)
    file_name = os.path.basename(model_path)
    backup_path = os.path.join(dir_path, 'old_' + file_name)

    print(f"\n💾 正在备份原文件为: {backup_path}")
    shutil.copy2(model_path, backup_path)

    print(f"💾 正在保存重命名后的模型到: {model_path}")
    torch.save(new_checkpoint, model_path)

    print("\n🎉 操作完成！")


# ==============================
# 使用示例
# ==============================
if __name__ == "__main__":
    model_path = "/GPFS/data/changxingliu/HEAL/opencood/logs/homo_czc/m7/net_epoch_bestval_at19.pth"  # ← 修改为你的路径

    # 定义任意前缀映射（支持多级）
    prefix_mapping = {
        'pyramid_backbone_m7': 'fusion_backbone',
        'shrink_conv_m7': 'shrink_conv',
        'cls_head_m7': 'cls_head',
        'dir_head_m7': 'dir_head',
        'reg_head_m7': 'reg_head'
    }

    rename_model_arbitrary_prefix(model_path, prefix_mapping)