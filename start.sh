#!/bin/bash

export PORT=${PORT:-8080}
export HOST=${HOST:-0.0.0.0}
export PYTHONUNBUFFERED=1

mkdir -p /app/logs

echo "Starting Gradio server..."
exec python huggingface_deploy.py
