# AI Trading System

A modular AI trading application that integrates news analysis, technical indicators, and sentiment data to generate actionable trading signals. The system serves predictions using a Hugging Face model and automates trades by sending webhooks to [TradersPost](https://traderspost.io/).

---

## 🚀 Features

- **Multi-modal data ingestion:** News, social sentiment, technical analysis, institutional flows.
- **Advanced ML models:** BERT/FinBERT for NLP, LSTM/Transformer for time series, ensemble for final signal.
- **Automated trading:** Sends signals to TradersPost via webhooks every 15 minutes.
- **Cloud-native:** Deployable to Google Cloud Run. Model hosting on Hugging Face Spaces.
- **Risk management:** Customizable position sizing, stop-loss, take-profit, and exposure limits.
- **Monitoring:** Real-time P&L, Sharpe ratio, model drift detection.

---

## 🏗️ Project Structure

```
.
├── app.py                # Flask/Gunicorn API server for inference and webhook
├── config.py             # Configuration for API keys, endpoints, and settings
├── trading/              # Trading logic, risk management, webhook integration
├── data/                 # Data collection/preprocessing scripts
├── model/                # Model architecture, training, and inference pipeline
├── requirements.txt      # Python dependencies
├── Dockerfile            # Container spec for Cloud Run
├── start.sh              # Entrypoint script (Gunicorn or Flask)
└── README.md
```

---

## ⚡ Quickstart

### 1. Clone the repo

```sh
git clone https://github.com/bentzbk/ai_trading_system.git
cd ai_trading_system
```

### 2. Install dependencies

```sh
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Set environment variables

Create a `.env` file or set these in your deployment:

```dotenv
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
NEWS_API_KEY=your_news_api_key
REDDIT_CLIENT_ID=your_reddit_id
REDDIT_CLIENT_SECRET=your_reddit_secret
REDDIT_USER_AGENT=your_bot_agent
TRADERSPOST_WEBHOOK_URL=https://api.traderspost.io/webhook/your_webhook
HF_MODEL_REPO=https://huggingface.co/spaces/bentzbk/wto
# ...add other secrets as needed
```

### 4. Run locally

```sh
python app.py
# or, for production
./start.sh
```
The API will run on [http://localhost:8080](http://localhost:8080)

---

## ☁️ Deploy to Google Cloud Run

### 1. Build and push the Docker image

```sh
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/ai_trading_system
```

### 2. Deploy to Cloud Run

```sh
gcloud run deploy ai-trading-system \
  --image gcr.io/YOUR_PROJECT_ID/ai_trading_system \
  --platform managed \
  --region us-east4 \
  --allow-unauthenticated \
  --port 8080
```

Set environment variables in the Cloud Run console or with `--set-env-vars`.

---

## 🔄 Automated Trading Loop

Every 15 minutes:
1. Fetches fresh market, news, and sentiment data.
2. Runs inference with the Hugging Face model.
3. Applies risk management and position sizing.
4. Formats signal as TradersPost webhook.
5. Sends trade instruction to TradersPost.
6. Logs trade and performance metrics.

---

## 🔗 Example TradersPost Webhook

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "action": "buy",
  "ticker": "AAPL",
  "quantity": 100,
  "type": "market",
  "confidence": 0.85,
  "stop_loss": 150.00,
  "take_profit": 160.00
}
```

---

## 📝 Model Deployment (Hugging Face)

- Fork and adapt [`bentzbk/wto`](https://huggingface.co/spaces/bentzbk/wto).
- Upload your trained model files.
- Modify the inference pipeline as needed.
- The app expects to call the Hugging Face API for predictions.

---

## 📊 Monitoring & Optimization

- Real-time P&L, Sharpe, drawdown, win rate.
- Model drift detection and retraining triggers.
- Performance logs are kept for all trades and model outputs.

---

## 🛡️ Risk Management

- Max 20% portfolio exposure
- 2% stop loss, 4% take profit
- Max 3 positions in the same sector
- Market regime filters (no new trades in high volatility)

---

## ⚠️ Risk Disclaimer

> Automated trading involves significant risks.  
> Always start with paper trading, use small positions, monitor closely, and be ready to intervene.

---

## 📚 References

- [Hugging Face Spaces - bentzbk/wto](https://huggingface.co/spaces/bentzbk/wto)
- [TradersPost Webhook Docs](https://traderspost.io/docs/webhooks)
- [Google Cloud Run](https://cloud.google.com/run/docs/quickstarts/build-and-deploy)

---

## 🛠️ Troubleshooting

- Check Cloud Run logs if the service fails to start.
- Ensure all secrets/API keys are set as environment variables.
- The inference API must be listening on `0.0.0.0:8080` for Cloud Run compatibility.
- Use `docker run -p 8080:8080 ai_trading_system` to test locally.

---

## ✅ Next Steps

1. Set up API keys and environment variables.
2. Deploy the app to Cloud Run or run locally.
3. Connect your TradersPost account via webhook.
4. Monitor, iterate, and optimize your trading strategy!
