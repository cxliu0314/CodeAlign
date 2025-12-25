import torch
from collections import defaultdict

def compare_models_deep(
    model_path1,
    model_path2,
    param_key_mapping=None
):
    """
    比较两个模型的参数是否相同。
    
    行为由 param_key_mapping 控制：
      - 若为 None 或 {}：直接按原始 key 一对一比较（要求结构完全一致）。
      - 若为非空 dict：仅比较指定映射的参数对，忽略其余参数。
    
    Args:
        model_path1: 第一个模型路径或 state_dict
        model_path2: 第二个模型路径或 state_dict
        param_key_mapping (dict or None): 
            格式：{ model2_full_prefix: model1_full_prefix }
            例如：{'backend_models.m1.cls_head': 'cls_head'}
            若为 None 或 {}，则直接比较原始 state_dict（key 必须完全一致）。
    
    Returns:
        dict: 比较结果
    """
    def load_state_dict(path):
        if isinstance(path, dict):
            return path
        checkpoint = torch.load(path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            return checkpoint['state_dict']
        elif hasattr(checkpoint, 'state_dict'):
            return checkpoint.state_dict()
        else:
            return checkpoint

    state_dict1 = load_state_dict(model_path1)
    state_dict2 = load_state_dict(model_path2)

    # 判断是否启用“全量直接比较”模式
    use_direct_compare = (param_key_mapping is None) or (len(param_key_mapping) == 0)

    if use_direct_compare:
        # 直接使用原始 state_dict，不做映射
        mapped_state_dict1 = state_dict1
        mapped_state_dict2 = state_dict2
    else:
        # 仅比较指定映射的参数
        mapped_state_dict1 = {}
        mapped_state_dict2 = {}

        # 按前缀长度降序排序，避免短前缀误匹配
        sorted_mappings = sorted(param_key_mapping.items(), key=lambda x: len(x[0]), reverse=True)

        for key2, param2 in state_dict2.items():
            for src_prefix, dst_prefix in sorted_mappings:
                if key2 == src_prefix:
                    key1 = dst_prefix
                    if key1 in state_dict1:
                        mapped_state_dict1[key1] = state_dict1[key1]
                        mapped_state_dict2[key1] = param2
                    break
                elif key2.startswith(src_prefix + '.'):
                    suffix = key2[len(src_prefix) + 1:]
                    key1 = f"{dst_prefix}.{suffix}"
                    if key1 in state_dict1:
                        mapped_state_dict1[key1] = state_dict1[key1]
                        mapped_state_dict2[key1] = param2
                    break

    # 如果映射后无参数（且非 direct 模式），返回提示
    if not use_direct_compare and not mapped_state_dict1:
        return {"_summary": {"status": "NO_MAPPED_PARAMS", "details": "未找到任何匹配的映射参数"}}

    # 按第一级模块分组（用于结构化输出）
    def group_by_first_level(state_dict):
        groups = defaultdict(dict)
        for key, param in state_dict.items():
            first_level = key.split('.')[0]
            groups[first_level][key] = param
        return groups

    groups1 = group_by_first_level(mapped_state_dict1)
    groups2 = group_by_first_level(mapped_state_dict2)

    print(f"模型1参数: {groups1.keys()}")
    print(f"模型2参数: {groups2.keys()}")

    all_modules = set(groups1.keys()) | set(groups2.keys())
    comparison_results = {}

    for module in sorted(all_modules):
        if module not in groups1:
            comparison_results[module] = {
                'status': 'MISSING_IN_MODEL1',
                'details': f'模块 {module} 在模型1中不存在'
            }
            continue

        if module not in groups2:
            comparison_results[module] = {
                'status': 'MISSING_IN_MODEL2',
                'details': f'模块 {module} 在模型2中不存在'
            }
            continue

        module_params1 = groups1[module]
        module_params2 = groups2[module]

        param_keys1 = set(module_params1.keys())
        param_keys2 = set(module_params2.keys())

        if param_keys1 != param_keys2:
            missing_in_1 = param_keys2 - param_keys1
            missing_in_2 = param_keys1 - param_keys2
            comparison_results[module] = {
                'status': 'PARAM_KEYS_MISMATCH',
                'details': {
                    'missing_in_model1': list(missing_in_1),
                    'missing_in_model2': list(missing_in_2)
                }
            }
            continue

        param_comparison = {}
        all_equal = True

        for param_key in sorted(param_keys1):
            param1 = module_params1[param_key]
            param2 = module_params2[param_key]

            if param1.shape != param2.shape:
                param_comparison[param_key] = {
                    'equal': False,
                    'reason': f'形状不同: {param1.shape} vs {param2.shape}',
                    'dtype': str(param1.dtype)
                }
                all_equal = False
            elif not torch.equal(param1, param2):
                diff = torch.abs(param1 - param2)
                diff_info = {
                    'equal': False,
                    'reason': '参数值不同',
                    'dtype': str(param1.dtype)
                }
                if param1.is_floating_point():
                    diff_info['max_diff'] = torch.max(diff).item()
                    diff_info['mean_diff'] = torch.mean(diff).item()
                else:
                    diff_info['max_diff'] = torch.max(diff).item()
                    diff_info['mean_diff'] = 'N/A (integer type)'
                    diff_info['mean_diff_float'] = torch.mean(diff.float()).item()
                param_comparison[param_key] = diff_info
                all_equal = False
            else:
                param_comparison[param_key] = {
                    'equal': True,
                    'reason': '参数完全相同',
                    'dtype': str(param1.dtype)
                }

        if all_equal:
            comparison_results[module] = {
                'status': 'IDENTICAL',
                'details': f'模块 {module} 的所有参数完全相同'
            }
        else:
            comparison_results[module] = {
                'status': 'DIFFERENT',
                'details': param_comparison
            }

    return comparison_results


def print_comparison_results(results):
    if "_summary" in results:
        print("⚠️  比较结果:", results["_summary"]["details"])
        return

    print("模型比较结果:")
    print("=" * 80)
    
    identical_count = 0
    different_count = 0
    missing_count = 0
    
    for module, result in results.items():
        status = result['status']
        
        if status == 'IDENTICAL':
            print(f"✅ {module:25s} : 完全相同")
            identical_count += 1
        elif status == 'DIFFERENT':
            print(f"❌ {module:25s} : 参数不同")
            different_count += 1
            details = result['details']
            for param_key, param_info in details.items():
                if not param_info['equal']:
                    dtype_info = f" ({param_info['dtype']})"
                    print(f"     └── {param_key:35s}: {param_info['reason']}{dtype_info}")
                    if 'max_diff' in param_info:
                        max_diff = param_info['max_diff']
                        mean_diff = param_info.get('mean_diff', 'N/A')
                        if isinstance(mean_diff, str) and 'N/A' in mean_diff:
                            mean_diff_val = param_info.get('mean_diff_float', 'N/A')
                            print(f"          最大差异: {max_diff}, 平均差异 (float): {mean_diff_val}")
                        else:
                            print(f"          最大差异: {max_diff:.6e}, 平均差异: {mean_diff:.6e}")
        elif status.startswith('MISSING'):
            print(f"⚠️  {module:25s} : {result['details']}")
            missing_count += 1
        elif status == 'PARAM_KEYS_MISMATCH':
            print(f"🔸 {module:25s} : 参数键不匹配")
            different_count += 1
            details = result['details']
            if details['missing_in_model1']:
                print(f"     └── 模型1缺少: {details['missing_in_model1']}")
            if details['missing_in_model2']:
                print(f"     └── 模型2缺少: {details['missing_in_model2']}")
    
    print("=" * 80)
    print(f"统计: 相同模块 {identical_count}个, 不同模块 {different_count}个, 缺失模块 {missing_count}个")


# 使用示例
if __name__ == "__main__":
    model1_path = "/GPFS/data/changxingliu/HEAL/opencood/logs/_refine/single_group_formation/m1_16_fixenc_aligner_fixback_2025_10_27_21_31_53/net_epoch_bestval_at42.pth"
    model2_path = "/GPFS/data/changxingliu/HEAL/opencood/logs/_refine/inter_group_test/C2C_m2_to_m1_coop_lr0001_aligner_2025_11_04_21_50_42/net_epoch_bestval_at18.pth"

    # # ✅ 方式1: 精确映射（只比指定参数）
    # param_key_mapping = {
    #     'backend_models.m1.codebook': 'codebook',
    #     'backend_models.m1.fusion': 'fusion_backbone',
    #     'backend_models.m1.shrink_conv': 'shrink_conv',
    #     'backend_models.m1.cls_head': 'cls_head',
    #     'backend_models.m1.reg_head': 'reg_head',
    #     'backend_models.m1.dir_head': 'dir_head',
    # }

    # ✅ 方式2: 全量直接比较（结构必须一致）
    param_key_mapping = {}  # 或者设为 None

    results = compare_models_deep(
        model_path1=model1_path,
        model_path2=model2_path,
        param_key_mapping=param_key_mapping
    )
    print_comparison_results(results)