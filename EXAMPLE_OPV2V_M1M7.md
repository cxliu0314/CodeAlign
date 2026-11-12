# OPV2V LiDAR + camera example

The [CodeAlign-OPV2V-m1m7-FCF-v1 model](https://huggingface.co/cxliu0314/CodeAlign-OPV2V-m1m7-FCF-v1) includes five checkpoints and the configuration for m1 (LiDAR) + m7 (camera) inference.

## Download

```bash
git clone https://github.com/cxliu0314/CodeAlign.git
cd CodeAlign
mkdir -p examples
curl -fL https://huggingface.co/cxliu0314/CodeAlign-OPV2V-m1m7-FCF-v1/resolve/main/codealign-opv2v-m1m7-fcf-v1.tar.gz | tar -xz -C examples
```

## Inference

Install the [environment](examples/opv2v_m1_m7/ENVIRONMENT.md) and prepare OPV2V with camera/depth data.

```bash
python opencood/utils/setup.py build_ext --inplace
CUDA_VISIBLE_DEVICES=0 bash examples/opv2v_m1_m7/run_inference.sh \
  --data-root /path/to/OPV2V --max-samples 4
```

Use `--max-samples 0` for the full test set. Outputs are saved to `examples/opv2v_m1_m7/results/`; set `--output-dir` to choose another directory.

Weights and original example resources use [Apache-2.0](examples/opv2v_m1_m7/LICENSE); see [LICENSE_NOTICE.md](examples/opv2v_m1_m7/LICENSE_NOTICE.md).
