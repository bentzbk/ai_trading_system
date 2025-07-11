import os
import sys
import logging
from flask import Flask, jsonify, request
from datetime import datetime
import traceback
from config import Config

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class TradingSystem:
    def __init__(self):
        self.initialized = False
        self.model = None
        self.setup_system()
    
    def setup_system(self):
        """Initialize the trading system"""
        try:
            logger.info("Initializing trading system...")
            
            # Validate configuration
            Config.validate()
            
            # Initialize model (using local or mock model for now)
            if Config.USE_LOCAL_MODEL:
                self.initialize_local_model()
            else:
                self.initialize_hf_model()
            
            self.initialized = True
            logger.info("Trading system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize trading system: {str(e)}")
            logger.error(traceback.format_exc())
            # Don't raise exception to allow container to start
            self.initialized = False
    
    def initialize_local_model(self):
        """Initialize with a local/mock model"""
        logger.info("Using local model configuration")
        # For now, use a simple mock model
        self.model = {"type": "mock", "status": "ready"}
    
    def initialize_hf_model(self):
        """Initialize with Hugging Face model"""
        try:
            from transformers import AutoTokenizer, AutoModel
            
            model_repo = getattr(Config, 'HF_MODEL_REPO', None)
            if not model_repo:
                raise ValueError("HF_MODEL_REPO not configured")
            
            logger.info(f"Loading model from {model_repo}")
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_repo)
            model = AutoModel.from_pretrained(model_repo)
            
            self.model = {
                "tokenizer": tokenizer,
                "model": model,
                "type": "huggingface",
                "status": "ready"
            }
            
            logger.info("Hugging Face model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Hugging Face model: {str(e)}")
            raise
    
    def predict(self, data):
        """Make trading predictions"""
        if not self.initialized or not self.model:
            return {"error": "System not initialized"}
        
        # Mock prediction for now
        return {
            "prediction": "hold",
            "confidence": 0.75,
            "timestamp": datetime.now().isoformat()
        }

# Initialize trading system
trading_system = TradingSystem()

@app.route('/')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "system_initialized": trading_system.initialized,
        "port": Config.PORT
    })

@app.route('/health')
def health():
    """Kubernetes/Cloud Run health check"""
    return jsonify({"status": "ok"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    """Trading prediction endpoint"""
    try:
        if not trading_system.initialized:
            return jsonify({"error": "Trading system not initialized"}), 500
        
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        result = trading_system.predict(data)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/status')
def status():
    """System status endpoint"""
    return jsonify({
        "system_initialized": trading_system.initialized,
        "model_status": trading_system.model.get("status", "unknown") if trading_system.model else "not_loaded",
        "config": {
            "port": Config.PORT,
            "host": Config.HOST,
            "use_local_model": Config.USE_LOCAL_MODEL
        }
    })

@app.errorhandler(Exception)
def handle_exception(e):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(e)}")
    logger.error(traceback.format_exc())
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    try:
        logger.info(f"Starting server on {Config.HOST}:{Config.PORT}")
        app.run(host=Config.HOST, port=Config.PORT, debug=False)
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        sys.exit(1)
