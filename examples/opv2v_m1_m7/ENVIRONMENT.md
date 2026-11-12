# Inference environment

Use Linux x86-64 with an NVIDIA CUDA GPU. Main versions: Python 3.7.12, PyTorch 1.10.1, torchvision 0.11.2, CUDA runtime 11.3, NumPy 1.21.6 and spconv 2.3.6 (`spconv-cu116`).

```bash
conda env create -f examples/opv2v_m1_m7/environment.yml
conda activate codealign-opv2v
python -m pip install -r examples/opv2v_m1_m7/requirements-inference.txt
python -m pip install git+https://github.com/klintan/pypcd.git@796e545df2ee5e7836df6fe2f274aae3428d1ac2
```

Install `torch-scatter==2.1.1` using a binary matching Python/PyTorch/CUDA, or build it with a CUDA 11.x toolkit compatible with PyTorch:

```bash
export CUDA_HOME=/path/to/cuda-11.3
export PATH="$CUDA_HOME/bin:$PATH"
python -m pip install --no-build-isolation --no-deps torch-scatter==2.1.1
```

From the CodeAlign repository root, build the local Cython extension:

```bash
python opencood/utils/setup.py build_ext --inplace
```

Prepare OPV2V with camera/depth data, including `additional-001.zip`, and pass the directory containing `test/` as `--data-root`.
