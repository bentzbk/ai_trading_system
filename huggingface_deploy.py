from huggingface_hub import HfApi, Repository, upload_file, login
import torch
import json
import os
from model import TradingTransformer, TradingModelTrainer

class HuggingFaceDeployer:
    def __init__(self, repo_name="bentzbk/woof_trade_organziation"):
        self.repo_name = repo_name
        self.api = HfApi()

    def create_model_card(self):
        """Create model card for Hugging Face"""
        model_card = """
---
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
        return model_card

    def create_gradio_app(self):
        """Create Gradio app for Hugging Face Space"""
        app_code = """
import gradio as gr
import torch
import numpy as np
import pandas as pd
from model import TradingTransformer, TradingModelTrainer
import yfinance as yf
import json

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

# Create Gradio interface
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
"""
        return app_code

    def deploy_to_huggingface(self, model_path):
        """Deploy model to Hugging Face"""
        try:
            # Login to Hugging Face
            login(token=os.getenv('HUGGINGFACE_TOKEN'))

            # Create repository
            repo_url = self.api.create_repo(
                repo_id=self.repo_name,
                token=os.getenv('HUGGINGFACE_TOKEN'),
                exist_ok=True
            )

            # Upload model file
            upload_file(
                path_or_fileobj=model_path,
                path_in_repo="trading_model_v1.pth",
                repo_id=self.repo_name,
                token=os.getenv('HUGGINGFACE_TOKEN')
            )

            # Upload model code
            with open("model.py", "r") as f:
                model_code = f.read()

            with open("temp_model.py", "w") as f:
                f.write(model_code)

            upload_file(
                path_or_fileobj="temp_model.py",
                path_in_repo="model.py",
                repo_id=self.repo_name,
                token=os.getenv('HUGGINGFACE_TOKEN')
            )

            # Create and upload app
            app_code = self.create_gradio_app()
            with open("temp_app.py", "w") as f:
                f.write(app_code)

            upload_file(
                path_or_fileobj="temp_app.py",
                path_in_repo="app.py",
                repo_id=self.repo_name,
                token=os.getenv('HUGGINGFACE_TOKEN')
            )

            # Create and upload model card
            model_card = self.create_model_card()
            with open("temp_readme.md", "w") as f:
                f.write(model_card)

            upload_file(
                path_or_fileobj="temp_readme.md",
                path_in_repo="README.md",
                repo_id=self.repo_name,
                token=os.getenv('HUGGINGFACE_TOKEN')
            )

            # Clean up temp files
            os.remove("temp_model.py")
            os.remove("temp_app.py")
            os.remove("temp_readme.md")

            print(f"Model deployed successfully to: https://huggingface.co/spaces/{self.repo_name}")

        except Exception as e:
            print(f"Deployment failed: {e}")

# Deploy the model
if __name__ == "__main__":
    deployer = HuggingFaceDeployer("bentzbk/woof_trade_organziation")
    deployer.deploy_to_huggingface("trading_model_v1.pth")
