"""
Deployment helper script - Ensures all required models and indices are available
Runs during build process on Render
"""
import os
import sys
from pathlib import Path

def ensure_directories():
    """Create required directories"""
    directories = [
        'saved_models',
        'faiss_index',
        'media/uploads',
        'media/gradcam_outputs',
        'reports'
    ]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    print(f"✅ Created {len(directories)} directories")

def check_models():
    """Check if required models exist"""
    required_files = {
        'saved_models/resnet50_classifier.pth': 'ResNet50 Classifier',
        'faiss_index/defect_index.faiss': 'FAISS Index',
        'faiss_index/index_metadata.json': 'FAISS Metadata'
    }
    
    missing = []
    for file_path, description in required_files.items():
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"✅ {description}: {file_path} ({size_mb:.1f} MB)")
        else:
            missing.append((file_path, description))
            print(f"⚠️  {description}: {file_path} (NOT FOUND)")
    
    if missing:
        print("\n⚠️  WARNING: Missing required files:")
        for file_path, description in missing:
            print(f"   - {description} ({file_path})")
        print("\n   These files should be included via Git LFS.")
        print("   If deploying without LFS, ensure they're present in the repository.")
    
    return len(missing) == 0

def verify_deployment():
    """Verify deployment configuration"""
    print("\n📋 Deployment Verification:")
    print(f"   Python: {sys.version.split()[0]}")
    print(f"   Django Settings: {os.getenv('DJANGO_SETTINGS_MODULE', 'config.settings')}")
    print(f"   Debug Mode: {os.getenv('DJANGO_DEBUG', 'False')}")
    print(f"   Deployment Mode: {os.getenv('DEPLOYMENT_MODE', 'development')}")
    
    # Check storage space
    try:
        import shutil
        stat = shutil.disk_usage('.')
        used_gb = stat.used / (1024**3)
        total_gb = stat.total / (1024**3)
        free_gb = stat.free / (1024**3)
        print(f"\n💾 Storage:")
        print(f"   Total: {total_gb:.1f} GB")
        print(f"   Used: {used_gb:.1f} GB")
        print(f"   Free: {free_gb:.1f} GB")
    except:
        pass

if __name__ == '__main__':
    print("🚀 Starting deployment checks...\n")
    
    ensure_directories()
    print()
    models_ok = check_models()
    verify_deployment()
    
    if models_ok:
        print("\n✅ All required files present - deployment ready!")
        sys.exit(0)
    else:
        print("\n⚠️  Some files missing - check Git LFS configuration")
        sys.exit(0)  # Don't fail the build, warn instead
