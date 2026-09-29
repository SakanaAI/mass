FROM python:3.11-slim-bookworm
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git curl libgomp1 libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN pip install "numpy<2.1" pandas scipy scikit-learn lightgbm xgboost catboost statsmodels matplotlib seaborn tqdm joblib optuna pyarrow openpyxl pillow opencv-python-headless \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install transformers sentencepiece datasets accelerate evaluate gymnasium networkx sympy tabulate
RUN mkdir -p /workspace/code /workspace/results && chmod -R a+rwX /workspace
WORKDIR /workspace
