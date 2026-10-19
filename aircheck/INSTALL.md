# Installation

To run the AIRCHECK notebooks:

1. Install the repository as described in the main [README](../README.md) (conda environment, `pip install -e .['dev']`, then `pip install -r requirements.txt`). If `pytorch-fast-transformers` fails to build, follow the [troubleshooting section](#troubleshooting-pytorch-fast-transformers) below.
2. Install the extra packages the AIRCHECK notebooks use:
   ```bash
   pip install -r aircheck/requirements-aircheck.txt
   ```
3. Open a notebook in `aircheck/notebooks/` with that environment as the kernel, and run it from that folder (the default paths, like `../sample_data/train.parquet`, are relative to it).

## Troubleshooting: `pytorch-fast-transformers`

On some Linux machines, `pip install -r requirements.txt` from the main [README](../README.md) fails while building `pytorch-fast-transformers==0.4.0`. This page explains why and how to fix it.

Don't skip the package. `bmfm_sm/predictive/modules/text_models.py` imports `fast_transformers`, so the text view (and the multi-view models that use it) will not load without it.

### Who is affected

You will likely hit this if any of the following is true:

- Your system CUDA toolkit (`nvcc --version`) is **not 12.1**. For example, Google Cloud Vertex AI Workbench and Deep Learning VM images ship CUDA 13.0.
- Your system compiler is **GCC 12 or newer** (`gcc --version`), e.g. Debian 12 or Ubuntu 23.04+.

`pytorch-fast-transformers` has no prebuilt wheels, so pip compiles its C++/CUDA code locally. That compile has to use the same CUDA version as PyTorch (`torch==2.1.0` is built against CUDA 12.1) and a compiler that CUDA 12.1 supports.

### Symptoms

| Error in the pip output | Cause |
|---|---|
| `The detected CUDA version (13.0) mismatches the version that was used to compile PyTorch (12.1)` | System CUDA is not 12.1 |
| `pybind11/detail/../cast.h:45:120: error: expected template-name before '<' token` | GCC 12+ used with CUDA 12.1 |
| `fatal error: curand.h: No such file or directory` | cuRAND development headers missing |

`--no-build-isolation` (used in `requirements-mps.txt`) does not fix any of these.

### Fix

The fix installs CUDA 12.1 and GCC 11 **inside your conda environment**, so your system CUDA and compiler are left untouched. You do not need a GPU on the machine to do this.

Follow the main README up to and including `pip install -e .['dev']`, with your environment activated. Then, instead of `pip install -r requirements.txt`:

**1. Install the PyTorch Geometric extras (everything in `requirements.txt` except `pytorch-fast-transformers`):**

```bash
grep -v fast-transformers requirements.txt > /tmp/requirements-pyg.txt
pip install -r /tmp/requirements-pyg.txt
```

**2. Add the CUDA 12.1 compiler, cuRAND headers, and GCC 11 to the environment:**

```bash
conda install -y -c nvidia/label/cuda-12.1.0 cuda-nvcc cuda-cudart-dev libcurand-dev
conda install -y -c conda-forge gcc=11 gxx=11
```

**3. Build `pytorch-fast-transformers` against them:**

```bash
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export NVCC_PREPEND_FLAGS="-ccbin $CXX"
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9"

pip install --no-build-isolation --no-cache-dir pytorch-fast-transformers==0.4.0
```

Expect the build to take 15 to 30 minutes, with long stretches of no new output.

`TORCH_CUDA_ARCH_LIST` sets which GPUs the kernels are compiled for: V100 (7.0), T4 (7.5), A100 (8.0), A10/A6000 (8.6), and L4 (8.9). It is required on a machine with no GPU, because there is no GPU for the build to detect. Building for these architectures also means you will not need to reinstall if you add a GPU later. To build for a different GPU, add its [compute capability](https://developer.nvidia.com/cuda-gpus) to the list.

If you use micromamba instead of conda, replace `conda install` with `micromamba install`. `$CONDA_PREFIX` is set the same way once the environment is activated.

### Verify

```bash
python -c "from fast_transformers.causal_product import causal_product_cuda; import bmfm_sm.predictive.modules.text_models; print('OK')"
python -m unittest bmfm_sm.tests.all_tests
```

The first command should print `OK`. On a machine without a GPU, it confirms the CUDA kernels were built even though they can't run there. The unit tests run on CPU.

### Tested on

- Google Cloud Vertex AI Workbench, `e2-highmem-4` (no GPU), Debian 12, system CUDA 13.0, system GCC 12.2
- Python 3.11, `torch==2.1.0+cu121`
- Result: all 59 unit tests pass
