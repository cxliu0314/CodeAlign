import torch
from collections import defaultdict

def count_first_level_parameters(state_dict):
    """
    统计模型state_dict中第一层级的参数量
    
    Args:
        state_dict: 模型的状态字典，可以是checkpoint['state_dict']或model.state_dict()
    
    Returns:
        dict: 第一层级模块名到参数量的映射
        int: 总参数量
    """
    # 如果传入的是完整的checkpoint而不是state_dict
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    
    # 使用defaultdict来自动初始化计数器
    module_params = defaultdict(int)
    
    for key, param in state_dict.items():
        # 获取第一层级模块名
        first_level = key.split('.')[0]
        
        # 计算当前参数的参数量并累加
        param_count = param.numel()
        module_params[first_level] += param_count
    
    # 计算总参数量
    total_params = sum(module_params.values())
    
    return dict(module_params), total_params

def count_nested_parameters(state_dict, max_depth=2):
    """
    统计模型state_dict中嵌套层级的参数量
    
    Args:
        state_dict: 模型的状态字典
        max_depth: 最大分析深度，2表示分析到第二层
    
    Returns:
        dict: 嵌套结构的参数字典
        int: 总参数量
    """
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    
    nested_params = defaultdict(lambda: defaultdict(int))
    total_params = 0
    
    for key, param in state_dict.items():
        parts = key.split('.')
        param_count = param.numel()
        total_params += param_count
        
        # 构建嵌套结构
        current_dict = nested_params
        for i, part in enumerate(parts):
            if i < max_depth - 1:
                if part not in current_dict:
                    current_dict[part] = defaultdict(int)
                current_dict = current_dict[part]
            else:
                # 最后一层，累加参数量
                current_dict[part] += param_count
                break
    
    return dict(nested_params), total_params

def print_first_level_stats(module_params, total_params):
    """
    打印第一层级参数量统计结果
    
    Args:
        module_params: 第一层级参数字典
        total_params: 总参数量
    """
    print("第一层级模块参数量统计:")
    print("-" * 50)
    
    # 按参数量排序（从大到小）
    sorted_modules = sorted(module_params.items(), key=lambda x: x[1], reverse=True)
    
    for module_name, param_count in sorted_modules:
        percentage = (param_count / total_params) * 100
        print(f"{module_name:20s}: {param_count:>12,} 参数 ({percentage:6.2f}%)")
    
    print("-" * 50)
    print(f"{'总计':20s}: {total_params:>12,} 参数 (100.00%)")

def print_nested_stats(nested_params, total_params, indent=0):
    """
    打印嵌套层级参数量统计结果
    
    Args:
        nested_params: 嵌套参数字典
        total_params: 总参数量
        indent: 缩进级别，用于格式化输出
    """
    if indent == 0:
        print("嵌套层级模块参数量统计:")
        print("-" * 60)
    
    for key, value in sorted(nested_params.items(), 
                           key=lambda x: x[1] if isinstance(x[1], int) else sum_all_params(x[1]), 
                           reverse=True):
        if isinstance(value, int):
            # 叶子节点，显示参数量
            percentage = (value / total_params) * 100
            indent_str = "  " * indent
            print(f"{indent_str}{key:20s}: {value:>12,} 参数 ({percentage:6.2f}%)")
        else:
            # 非叶子节点，递归显示
            sub_total = sum_all_params(value)
            percentage = (sub_total / total_params) * 100
            indent_str = "  " * indent
            print(f"{indent_str}{key:20s}: {sub_total:>12,} 参数 ({percentage:6.2f}%)")
            print_nested_stats(value, total_params, indent + 1)
    
    if indent == 0:
        print("-" * 60)
        print(f"{'总计':20s}: {total_params:>12,} 参数 (100.00%)")

def sum_all_params(param_dict):
    """递归计算嵌套字典中所有参数的总和"""
    total = 0
    for value in param_dict.values():
        if isinstance(value, int):
            total += value
        else:
            total += sum_all_params(value)
    return total

# 使用示例
def analyze_model_parameters(checkpoint_path, depth=1):
    """
    完整的分析函数
    
    Args:
        checkpoint_path: checkpoint文件路径
        depth: 分析深度，1=仅第一层，2=包含第二层
    """
    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 获取state_dict
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif hasattr(checkpoint, 'state_dict'):
        state_dict = checkpoint.state_dict()
    else:
        state_dict = checkpoint
    
    if depth == 1:
        # 第一层分析
        module_params, total_params = count_first_level_parameters(state_dict)
        print_first_level_stats(module_params, total_params)
        return module_params, total_params
    else:
        # 嵌套分析
        nested_params, total_params = count_nested_parameters(state_dict, max_depth=depth)
        print_nested_stats(nested_params, total_params)
        return nested_params, total_params

# 更简洁的使用方式
def quick_analyze(state_dict, depth=1):
    """
    快速分析函数
    """
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    
    if depth == 1:
        module_params = {}
        for key, param in state_dict.items():
            first_level = key.split('.')[0]
            module_params[first_level] = module_params.get(first_level, 0) + param.numel()
        
        total_params = sum(module_params.values())
        
        # 打印结果
        print("模块\t\t参数量\t\t占比")
        print("-" * 40)
        for module, count in sorted(module_params.items(), key=lambda x: x[1], reverse=True):
            percent = (count / total_params) * 100
            print(f"{module:15s} {count:10,} {percent:8.2f}%")
        
        print("-" * 40)
        print(f"{'总计':15s} {total_params:10,} {100:8.2f}%")
        
        return module_params, total_params
    else:
        return analyze_model_parameters(state_dict, depth)

# 使用示例
if __name__ == "__main__":
    model = "/GPFS/data/changxingliu/HEAL/opencood/logs/_refine/translation/D2C_m27_to_m16_2025_11_16_23_24_41/net_epoch_bestval_at14.pth"
    # module_params, total = analyze_model_parameters(model, depth=1)
    module_params, total = analyze_model_parameters(model, depth=3)
    