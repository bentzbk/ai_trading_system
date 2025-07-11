import os
from typing import Optional

class Config:
    # Server Configuration
    PORT = int(os.getenv('PORT', 8080))
    HOST = os.getenv('HOST', '0.0.0.0')
    
    # Hugging Face Configuration
    # Option 1: Use a local model or disable model loading for now
    USE_LOCAL_MODEL = True
    MODEL_PATH = os.getenv('MODEL_PATH', './models/trading_model.pkl')
    
    # Option 2: Use an actual existing model from Hugging Face
    # Uncomment and use a real model if you have one
    # HF_MODEL_REPO = os.getenv('HF_MODEL_REPO', 'microsoft/DialoGPT-medium')
    # HF_TOKEN = os.getenv('HF_TOKEN', None)
    
    # Trading Configuration
    API_KEY = os.getenv('API_KEY', '')
    API_SECRET = os.getenv('API_SECRET', '')
    
    # Database Configuration
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///trading.db')
    
    # Logging Configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    @classmethod
    def validate(cls):
        """Validate configuration before starting the application"""
        if not cls.USE_LOCAL_MODEL and not hasattr(cls, 'HF_MODEL_REPO'):
            raise ValueError("Either USE_LOCAL_MODEL must be True or HF_MODEL_REPO must be set")
        
        return True
