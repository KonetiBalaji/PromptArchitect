#!/usr/bin/env python3
"""
Visionary Prompt Architect - Setup Script
Author: Balaji Koneti

Automated setup script for the Visionary Prompt Architect.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def create_env_file():
    """Create .env file from template"""
    env_content = """# Visionary Prompt Architect - Environment Configuration
# Author: Balaji Koneti

# Required API Keys
OPENAI_API_KEY=your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here
GOOGLE_API_KEY=your-google-api-key-here

# Security
SECRET_KEY=your-secret-key-here
JWT_SECRET=your-jwt-secret-here

# Database (Optional - defaults to SQLite)
DATABASE_URL=sqlite:///./prompt_architect.db

# Redis (Optional - defaults to in-memory)
REDIS_URL=redis://localhost:6379/0

# Application Settings
DEBUG=true
LOG_LEVEL=INFO
ENVIRONMENT=development

# Monitoring (Optional)
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true
"""
    
    env_file = Path(".env")
    if not env_file.exists():
        with open(env_file, "w") as f:
            f.write(env_content)
        print("✅ Created .env file")
    else:
        print("ℹ️  .env file already exists")


def install_dependencies():
    """Install Python dependencies"""
    print("📦 Installing dependencies...")
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True)
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False
    
    return True


def create_directories():
    """Create necessary directories"""
    directories = [
        "logs",
        "data",
        "cache",
        "monitoring/grafana/dashboards",
        "monitoring/grafana/datasources",
        "nginx/ssl"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Created necessary directories")


def setup_database():
    """Setup database (if using PostgreSQL)"""
    print("🗄️  Database setup...")
    
    # For now, we'll use SQLite by default
    # In production, you would set up PostgreSQL here
    print("ℹ️  Using SQLite for development (configure PostgreSQL for production)")


def main():
    """Main setup function"""
    print("🎯 Visionary Prompt Architect Setup")
    print("Author: Balaji Koneti")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 11):
        print("❌ Python 3.11+ is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Create directories
    create_directories()
    
    # Create .env file
    create_env_file()
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Setup database
    setup_database()
    
    print("\n🎉 Setup completed successfully!")
    print("\n📝 Next steps:")
    print("1. Edit .env file with your API keys")
    print("2. Run: python run.py dev")
    print("3. Visit: http://localhost:8000/docs")
    
    print("\n🔑 Required API Keys:")
    print("- OpenAI API Key: https://platform.openai.com/api-keys")
    print("- Anthropic API Key: https://console.anthropic.com/")
    print("- Google API Key: https://console.cloud.google.com/")


if __name__ == "__main__":
    main()
