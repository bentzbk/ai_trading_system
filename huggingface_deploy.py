"""
This script deploys your trading model, model code, Gradio app, and model card to a Hugging Face Space.
It is NOT intended to run as a web server, but as a one-off deployment tool.
Make sure you have set the HUGGINGFACE_TOKEN environment variable with write access to your target repo.
"""

from huggingface_hub import HfApi, upload_file, login
import os

# === CONFIGURATION ===
# Repo id must be in the form "spaces/username/reponame" for Spaces.
REPO_ID = "spaces/bentzbk/wto"
MODEL_LOCAL_PATH = "trading_model_v1.pth"  # Update if your model file is named differently
MODEL_FILE_IN_REPO = "trading_model_v1.pth"
MODEL_CODE_LOCAL_PATH = "model.py"
MODEL_CODE_FILE_IN_REPO = "model.py"
APP_FILE_IN_REPO = "app.py"
README_FILE_IN_REPO = "README.md"

# === MODEL CARD CONTENT ===
MODEL_CARD = """---
title: AI Trading Model
emoji: 📈
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 3.35.2
app_file: app.py
pinned: false
license: mit
---

# AI Trading Model

This is an AI-powered trading model that analyzes market data, news sentiment, and technical indicators to make trading decisions.

## Features
- Multi-modal analysis (price, volume, sentiment)
- Real-time predictions
- Risk management integration
- 15-minute trading signals

## Usage
The model returns:
- Action: buy/sell/hold
- Confidence score
- Position size recommendation
"""

# === GRADIO APP CONTENT ===
# You MUST have a model.py and trading_model_v1.pth in your repo for this app to run!
APP_CODE = '''
import gradio as gr
import torch
import numpy as np
import pandas as pd
from model import TradingTransformer, TradingModelTrainer
import yfinance as yf

# Load the model
trainer = TradingModelTrainer()
model = trainer.load_model("trading_model_v1.pth")

def predict_stock(symbol, period="1mo"):
    try:
        # Get stock data
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)

        if len(data) < 30:
            return "Error: Not enough data"

        # Get the last 30 days of data
        recent_data = data.tail(30)

        # Prepare features (simplified)
        features = []
        for _, row in recent_data.iterrows():
            features.append([
                row['Open'], row['High'], row['Low'],
                row['Close'], row['Volume']
            ])

        features = np.array(features).reshape(1, 30, 5)

        # Make prediction
        prediction = trainer.predict(model, features)

        action_map = {0: "BUY", 1: "SELL", 2: "HOLD"}
        action = action_map[prediction['action'][0]]
        confidence = prediction['confidence'][0][0]
        position_size = prediction['position_size'][0][0]

        return {
            "Symbol": symbol,
            "Action": action,
            "Confidence": f"{confidence:.2%}",
            "Position Size": f"{position_size:.2%}",
            "Current Price": f"${data['Close'].iloc[-1]:.2f}"
        }

    except Exception as e:
        return f"Error: {str(e)}"

iface = gr.Interface(
    fn=predict_stock,
    inputs=[
        gr.Textbox(label="Stock Symbol", placeholder="AAPL"),
        gr.Dropdown(choices=["1mo", "3mo", "6mo"], label="Period", value="1mo")
    ],
    outputs=gr.JSON(label="Prediction"),
    title="AI Trading Model",
    description="Enter a stock symbol to get AI-powered trading recommendations"
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=8080)
'''

class HuggingFaceDeployer:
    def __init__(self, repo_id):
        self.repo_id = repo_id
        self.api = HfApi()
        self.token = os.getenv('HUGGINGFACE_TOKEN')
        if not self.token:
            raise RuntimeError("Please set the HUGGINGFACE_TOKEN environment variable before running this script.")

    def login(self):
        print("Logging into Hugging Face Hub...")
        login(token=self.token)

    def ensure_repo(self):
        print(f"Ensuring repo {self.repo_id} exists (creating if needed)...")
        self.api.create_repo(
            repo_id=self.repo_id,
            token=self.token,
            repo_type="space",  # makes sure it's a Space, not a Model repo!
            exist_ok=True,
            space_sdk="gradio"
        )

    def upload_bytes(self, content: str, path_in_repo: str):
        print(f"Uploading in-memory content to {path_in_repo} ...")
        upload_file(
            path_or_fileobj=content.encode("utf-8"),
            path_in_repo=path_in_repo,
            repo_id=self.repo_id,
            token=self.token
        )

    def upload_local_file(self, local_path: str, path_in_repo: str):
        print(f"Uploading local file {local_path} to {path_in_repo} ...")
        if not os.path.isfile(local_path):
            raise FileNotFoundError(f"File {local_path} not found!")
        upload_file(
            path_or_fileobj=local_path,
            path_in_repo=path_in_repo,
            repo_id=self.repo_id,
            token=self.token
        )

    def deploy(self):
        self.login()
        self.ensure_repo()

        # Upload the model file
        self.upload_local_file(MODEL_LOCAL_PATH, MODEL_FILE_IN_REPO)

        # Upload model code
        self.upload_local_file(MODEL_CODE_LOCAL_PATH, MODEL_CODE_FILE_IN_REPO)

        # Upload app.py
        self.upload_bytes(APP_CODE, APP_FILE_IN_REPO)

        # Upload model card/README
        self.upload_bytes(MODEL_CARD, README_FILE_IN_REPO)

        print(f"\nModel and app deployed successfully to: https://huggingface.co/spaces/bentzbk/wto\n")

if __name__ == "__main__":
    deployer = HuggingFaceDeployer(REPO_ID)
    deployer.deploy()
