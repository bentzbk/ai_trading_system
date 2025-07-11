import os

class Config:
    # Server Configuration
    PORT = int(os.getenv('PORT', 8080))
    HOST = os.getenv('HOST', '0.0.0.0')
    
    # Hugging Face Configuration
    USE_LOCAL_MODEL = os.getenv('USE_LOCAL_MODEL', 'True').lower() == 'true'
    MODEL_PATH = os.getenv('MODEL_PATH', './models/trading_model.pkl')
    HF_MODEL_REPO = os.getenv('HF_MODEL_REPO', '')  # Leave blank unless using HF
    HF_TOKEN = os.getenv('HF_TOKEN', None)
    
    # Trading Configuration
    API_KEY = os.getenv('API_KEY', '')
    API_SECRET = os.getenv('API_SECRET', '')
    
    # Database Configuration
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///trading.db')
    
    # Logging Configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    @classmethod
    def validate(cls):
        if not cls.USE_LOCAL_MODEL and not cls.HF_MODEL_REPO:
            raise ValueError("Either USE_LOCAL_MODEL must be True or HF_MODEL_REPO must be set")
        return True
