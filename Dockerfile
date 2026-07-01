FROM python:3.11-slim

# System deps: Tesseract (+Khmer), OpenCV runtime libs, curl for uv
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr tesseract-ocr-khm libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

# uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app
# Dependency layer first for caching — mlx-lm is excluded by the darwin/arm64
# platform marker; Linux torch wheel includes CUDA support by default.
COPY pyproject.toml uv.lock ./
RUN uv sync --no-install-project

# App code
COPY src/ ./src/
COPY app.py lab.py ./
COPY fonts/ ./fonts/
RUN uv sync

EXPOSE 8501
# No SURYA_INFERENCE_BACKEND -> Surya uses the torch backend; device.py picks CUDA/CPU.
CMD ["uv", "run", "streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
