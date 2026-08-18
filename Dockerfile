# ---- Stage 1: build the React workspace -------------------------------------
# frontend/dist is gitignored, and webapp/api.py:730 only mounts /app when that
# directory exists — so without this stage the image would start cleanly and
# silently serve no primary UI.
FROM node:22-slim AS frontend-build
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- Stage 2: runtime --------------------------------------------------------
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
COPY webapp/ ./webapp/
COPY fonts/ ./fonts/
# Built React bundle from stage 1 — webapp/api.py mounts it at /app.
COPY --from=frontend-build /build/dist ./frontend/dist
RUN uv sync

EXPOSE 8600
# No SURYA_INFERENCE_BACKEND -> Surya uses the torch backend; device.py picks CUDA/CPU.
# NiceGUI's ui.run() already binds 0.0.0.0, so no host argument is needed here.
CMD ["uv", "run", "python", "-m", "webapp.main"]
