# CPU image for the MMELON prediction API (api.py).
# Build from the repo root:  docker build -f aircheck/serving/Dockerfile -t mmelon-api .
# Run:  docker run -p 8090:8080 -v <models folder>:/models -e MODELS_CONFIG=/models/models.json mmelon-api
# Models are not in the image; they are mounted at run time (on Cloud Run, from GCS).

FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

# g++ compiles pytorch-fast-transformers; libxrender1/libxext6 are needed by RDKit drawing (image view)
RUN apt-get update && apt-get install -y --no-install-recommends g++ libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# 1. CPU-only PyTorch (much smaller than the CUDA build)
RUN pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu

# 2. PyTorch Geometric extras, CPU wheels (the repo's requirements.txt uses the cu121 ones)
RUN pip install pyg_lib torch_scatter torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cpu.html

# 3. pytorch-fast-transformers (text view); compiled here, CPU only
RUN pip install --no-build-isolation setuptools==69.5.1 pytorch-fast-transformers==0.4.0

# 4. The MMELON code (bmfm_sm) and its dependencies from pyproject.toml. Editable, so its data files
#    (e.g. resources/vocab.txt) are used in place
WORKDIR /app
COPY pyproject.toml README.md ./
COPY bmfm_sm ./bmfm_sm
RUN SETUPTOOLS_SCM_PRETEND_VERSION=1.1 pip install -e .

# 5. API packages, LightGBM (same version as the training env), and the API code
COPY aircheck/serving/requirements-serving.txt ./serving/
RUN pip install -r serving/requirements-serving.txt lightgbm==4.7.0
COPY aircheck/serving/*.py ./serving/

WORKDIR /app/serving
ENV PORT=8080 MAX_SMILES=50 DEVICE=cpu
EXPOSE 8080
# Cloud Run sets PORT; MODELS_CONFIG must be given at run time
CMD exec uvicorn api:app --host 0.0.0.0 --port ${PORT}
