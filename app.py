import os
import gradio as gr
import numpy as np
import torch
from huggingface_hub import hf_hub_download
import yfinance as yf

from model import TradingTransformer, TradingModelTrainer

HF_MODEL_REPO = "bentzbk/woof_trade_organziation"
HF_MODEL_FILE = "trading_model_v1.pth"  # change if your model file is named differently

def load_model_from_hf():
    print("Downloading model from Hugging Face Hub...")
    try:
        model_path = hf_hub_download(
            repo_id=HF_MODEL_REPO,
            filename=HF_MODEL_FILE,
            token=os.getenv("HUGGINGFACE_TOKEN")
        )
        print(f"Model downloaded to {model_path}")
        trainer = TradingModelTrainer()
        model = trainer.load_model(model_path)
        return model, trainer
    except Exception as e:
        print(f"Failed to download or load model: {e}")
        raise

print("Starting application...")
model, trainer = load_model_from_hf()

def predict_stock(symbol, period="1mo"):
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        if len(data) < 30:
            return "Error: Not enough data"
        recent_data = data.tail(30)
        features = []
        for _, row in recent_data.iterrows():
            features.append([
                row['Open'], row['High'], row['Low'], 
                row['Close'], row['Volume']
            ])
        features = np.array(features).reshape(1, 30, 5)
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
        print(f"Error in predict_stock: {e}")
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
    print("Launching Gradio app on 0.0.0.0:8080")
    iface.launch(server_name="0.0.0.0", server_port=8080)
