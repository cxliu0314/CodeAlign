#!/usr/bin/env python3
"""Run the OPV2V m1+m7 pretrained example."""
import argparse
from contextlib import ExitStack
import os
from pathlib import Path
import sys
from unittest.mock import patch

import yaml

EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_DIR = EXAMPLE_DIR.parents[1]


def resolve_paths(value, data_root):
    if isinstance(value, dict):
        return {key: resolve_paths(item, data_root) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_paths(item, data_root) for item in value]
    if isinstance(value, str):
        if value.startswith('weights/'):
            path = (EXAMPLE_DIR / value).resolve()
            if EXAMPLE_DIR not in path.parents or not path.is_file():
                raise FileNotFoundError('Missing checkpoint: ' + value)
            return str(path)
        return value.replace('${OPV2V_ROOT}', str(data_root))
    return value


def link_path(path, target):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        if path.resolve() == target.resolve():
            return
        path.unlink()
    if path.exists():
        raise RuntimeError('Expected a symlink at ' + str(path))
    path.symlink_to(target, target_is_directory=target.is_dir())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True, help='OPV2V directory containing test/')
    parser.add_argument('--output-dir', type=Path, default=EXAMPLE_DIR / 'results')
    parser.add_argument('--max-samples', type=int, default=4, help='number of frames; 0 runs the full test set')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--no-vis', action='store_true')
    args = parser.parse_args()
    if args.max_samples < 0 or args.workers < 0:
        parser.error('--max-samples and --workers must be nonnegative')
    data_root = args.data_root.expanduser().resolve()
    if not (data_root / 'test').is_dir():
        parser.error('--data-root must contain test/')
    output_dir = args.output_dir.expanduser().resolve()
    runtime_dir = output_dir / 'runtime'
    runtime_dir.mkdir(parents=True, exist_ok=True)
    data_link = runtime_dir / 'dataset' / 'OPV2V'
    link_path(data_link, data_root)

    definition = resolve_paths(yaml.safe_load((EXAMPLE_DIR / 'modality_define.yaml').read_text()), data_link)
    definition_path = runtime_dir / 'modality_define.yaml'
    definition_path.write_text(yaml.safe_dump(definition, sort_keys=False))
    config = resolve_paths(yaml.safe_load((EXAMPLE_DIR / 'config.yaml').read_text()), data_link)
    config['heter']['definition_path'] = str(definition_path)
    config['heter']['assignment_path'] = str(EXAMPLE_DIR / 'opv2v_4modality.json')
    (runtime_dir / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    checkpoint = EXAMPLE_DIR / 'weights' / 'translator_m7_to_m1_epoch8.pth'
    if not checkpoint.is_file():
        raise FileNotFoundError('Missing checkpoint: ' + str(checkpoint))
    link_path(runtime_dir / 'net_epoch_bestval_at8.pth', checkpoint)

    sys.path.insert(0, str(REPO_DIR))
    os.chdir(str(REPO_DIR))
    import torch
    from torch.utils.data import Subset
    from efficientnet_pytorch import EfficientNet
    from opencood.tools import inference_heter_in_order as entry
    if not torch.cuda.is_available():
        raise RuntimeError('An NVIDIA CUDA GPU is required.')
    torch.manual_seed(303)
    torch.cuda.manual_seed_all(303)

    original_loader = entry.DataLoader

    def loader(dataset, *positional, **kwargs):
        if args.max_samples:
            dataset = Subset(dataset, range(min(len(dataset), args.max_samples)))
        kwargs['num_workers'] = args.workers
        return original_loader(dataset, *positional, **kwargs)

    options = argparse.Namespace(
        model_dir=str(runtime_dir), fusion_method='intermediate',
        output_dir=str(output_dir), save_vis_interval=40, no_vis=args.no_vis,
        legacy_model_init=False, save_npy=False, range='102.4,102.4',
        no_score=False, use_cav='[2]', lidar_degrade=False, note='', noise=0.0)
    with ExitStack() as stack:
        stack.enter_context(patch.object(entry, 'test_parser', return_value=options))
        stack.enter_context(patch.object(entry, 'DataLoader', side_effect=loader))
        # The bundled checkpoints supply all EfficientNet weights.
        stack.enter_context(patch.object(
            EfficientNet, 'from_pretrained',
            side_effect=lambda model_name, *a, **kw: EfficientNet.from_name(model_name)))
        entry.main()


if __name__ == '__main__':
    main()
