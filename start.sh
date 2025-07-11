#!/bin/bash

# Set environment variables
export PORT=${PORT:-8080}
export HOST=${HOST:-0.0.0.0}
export PYTHONUNBUFFERED=1

# Create log directory
mkdir -p /app/logs

# Start the application with gunicorn for production
if [ "$ENVIRONMENT" = "production" ]; then
    echo "Starting with gunicorn..."
    exec gunicorn --bind $HOST:$PORT --workers 2 --timeout 60 --keep-alive 2 --log-level info main:app
else
    echo "Starting with Flask development server..."
    exec python main.py
fi
