#!/usr/bin/env bash
# Activate Surya's llamacpp Metal backend for Apple Silicon.
# Source this file before running the pipeline or benchmark:
#   source setup-metal-macos.sh

export SURYA_INFERENCE_BACKEND=llamacpp
export SURYA_LLAMACPP_BINARY=$(which llama-server)
export LLAMA_CPP_NGL=99            # offload all model layers to Metal GPU
export SURYA_INFERENCE_PARALLEL=8  # concurrent page inferences

echo "Surya llamacpp Metal backend activated."
echo "  llama-server: $SURYA_LLAMACPP_BINARY"
