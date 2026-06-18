"""
Deployment configuration for CPU-based deployment
Handles conditional model loading based on environment
"""
import os

# CPU Deployment Flag
DEPLOYMENT_MODE = os.getenv('DEPLOYMENT_MODE', 'cpu')  # 'cpu' or 'local'

# Skip loading large ML models in CPU deployment if not available
SKIP_ML_MODELS = DEPLOYMENT_MODE == 'cpu'

# Model paths (optional, download at runtime if needed)
MODEL_PATHS = {
    'classifier': os.path.join('saved_models', 'resnet50_classifier.pth'),
    'faiss_index': os.path.join('faiss_index', 'defect_index.faiss'),
}

# Check if models exist before attempting to load
MODELS_AVAILABLE = {
    name: os.path.exists(path)
    for name, path in MODEL_PATHS.items()
}

# Render-specific configurations
if os.getenv('RENDER'):
    # On Render, set deployment mode
    DEPLOYMENT_MODE = 'cpu'
    # Use PostgreSQL from Render's DATABASE_URL
    import dj_database_url
    DATABASES = {
        'default': dj_database_url.config(
            default='sqlite:///db.sqlite3',
            conn_max_age=600
        )
    }
