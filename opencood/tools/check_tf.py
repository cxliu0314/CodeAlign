import re
import argparse
import os
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('log_directory', help='Path to the log directory containing log.log')
args = parser.parse_args()

base_dir = args.log_directory
log_file = os.path.join(base_dir, 'log.log')
output_dir = os.path.join(base_dir, 'trans')
writer = SummaryWriter(log_dir=output_dir)

with open(log_file, 'r') as f:
    lines = f.readlines()

# 匹配非 val 的主 loss 行
pattern_main = re.compile(r"\[epoch (\d+)]\[(\d+)/(\d+)](_single)?(?:val)?\s*\|\|(.+)")
# 匹配验证 loss 行
pattern_val_loss = re.compile(r"At epoch (\d+), the validation loss is ([\d\.eE+-]+)")
# 匹配特殊 loss（如 L2 Loss、Rec Loss）
pattern_special_loss = re.compile(r"([\w\s]+): ([\d\.eE+-]+)")

global_step = 0
i = 0
while i < len(lines):
    line = lines[i].strip()

    # 忽略 step 内 val 行
    if '[val' in line:
        i += 1
        continue

    # 处理 validation 总 loss
    match_val = pattern_val_loss.match(line)
    if match_val:
        epoch = int(match_val.group(1))
        val_loss = float(match_val.group(2))
        writer.add_scalar('Val_Loss', val_loss, epoch)
        i += 1
        print('Val_Loss', val_loss, epoch)
        continue

    # 匹配非-single主 loss 行
    match_main = pattern_main.match(line)
    if not match_main or match_main.group(4):  # 不是 _single，且匹配成功
        i += 1
        continue

    # ------ Step 1: 解析 multi（非 single）主 loss ------
    epoch = int(match_main.group(1))
    step = int(match_main.group(2))
    total_steps = int(match_main.group(3))
    main_losses = {}
    raw_losses = match_main.group(5)
    for item in raw_losses.split("||"):
        item = item.strip()
        if not item:
            continue
        try:
            name, value = item.split(":")
            main_losses[name.strip()] = float(value.strip())
        except ValueError:
            pass
    i += 1
    if i >= len(lines):
        break

    # ------ Step 2: 收集特殊 loss 行 ------
    special_losses = {}
    while i < len(lines):
        line_special = lines[i].strip()
        match_special = pattern_special_loss.match(line_special)
        if match_special:
            name = match_special.group(1).strip()
            value = float(match_special.group(2))
            special_losses[name] = value
            i += 1
        else:
            break  # 下一条不是特殊 loss，结束
    if i >= len(lines):
        break

    # ------ Step 3: 查找 single 行 ------
    line_single = lines[i].strip()
    match_single = pattern_main.match(line_single)
    single_losses = {}
    if match_single and match_single.group(4):  # 是 _single 行
        raw_losses_single = match_single.group(5)
        for item in raw_losses_single.split("||"):
            item = item.strip()
            if not item:
                continue
            try:
                name, value = item.split(":")
                single_losses[name.strip()] = float(value.strip())
            except ValueError:
                pass
        i += 1
    if i >= len(lines):
        break

    # ------ 汇总所有 loss 并写入 ------
    all_losses = {}

    # 非 single 的指定 loss 项
    for name in ['Conf Loss', 'Loc Loss', 'Dir Loss', 'Depth Loss', 'Codebook Loss']:
        if name in main_losses:
            all_losses[name.replace(" ", "_")] = main_losses[name]

    # 加上 single 中的 Pyramid Loss（如果有）
    pyramid = single_losses.get('Pyramid Loss', 0.0)
    all_losses['Pyramid_Loss'] = pyramid

    # 合成 Total Loss：multi Loss + single Loss
    total = main_losses.get('Loss', 0.0) + single_losses.get('Loss', 0.0)
    all_losses['Total_Loss'] = total

    # 特殊 loss
    for name, value in special_losses.items():
        all_losses[name.replace(" ", "_")] = value

    # 写入 TensorBoard
    for tag, val in all_losses.items():
        writer.add_scalar(tag, val, global_step)

    global_step += 1
    if global_step % 1000 == 0: print(f"Processed epoch {epoch}, step {step}/{total_steps}, global step {global_step}, losses: {all_losses}")
    if i >= len(lines):
        break

writer.close()
print("日志转换完成，输出保存在:", output_dir)