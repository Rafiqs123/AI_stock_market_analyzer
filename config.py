import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Application settings
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    SECRET_KEY = os.getenv('SECRET_KEY', 'tajny-klucz-domyslny')
    
    # Flask settings
    CACHE_TYPE = 'simple'
    CACHE_DEFAULT_TIMEOUT = 300
    
    # API limits
    API_RATE_LIMIT = 2  # wywołania na sekundę
    API_RATE_LIMIT_PERIOD = 1  # w sekundach
    
    # Model settings
    MODEL_PATH = os.getenv('MODEL_PATH', './models')
    DEFAULT_MODEL = "microsoft/phi-2"
    
    # Monitoring
    SENTRY_DSN = os.getenv('SENTRY_DSN')
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
