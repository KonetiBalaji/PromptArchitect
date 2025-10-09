#!/usr/bin/env python3
"""
Visionary Prompt Architect - Run Script
Author: Balaji Koneti

Simple script to run the application with different configurations.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def check_requirements():
    """Check if all requirements are met"""
    print("🔍 Checking requirements...")
    
    # Check Python version
    if sys.version_info < (3, 11):
        print("❌ Python 3.11+ is required")
        return False
    
    # Check if virtual environment is activated
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Virtual environment not detected. Consider using one.")
    
    # Check if requirements are installed
    try:
        import fastapi
        import uvicorn
        print("✅ Core dependencies found")
    except ImportError:
        print("❌ Dependencies not installed. Run: pip install -r requirements.txt")
        return False
    
    return True


def check_env_file():
    """Check if .env file exists and has required keys"""
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env file not found. Creating template...")
        create_env_template()
        return False
    
    # Check for required keys
    required_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "SECRET_KEY"]
    missing_keys = []
    
    with open(env_file, 'r') as f:
        content = f.read()
        for key in required_keys:
            if f"{key}=" not in content or f"{key}=your-" in content:
                missing_keys.append(key)
    
    if missing_keys:
        print(f"❌ Missing or incomplete API keys: {', '.join(missing_keys)}")
        print("Please edit .env file with your actual API keys")
        return False
    
    print("✅ Environment configuration looks good")
    return True


def create_env_template():
    """Create .env template file"""
    env_template = """# Visionary Prompt Architect - Environment Configuration
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
    
    with open(".env", "w") as f:
        f.write(env_template)
    
    print("✅ Created .env template file")
    print("📝 Please edit .env file with your actual API keys")


def run_development():
    """Run in development mode"""
    print("🚀 Starting Visionary Prompt Architect in development mode...")
    
    if not check_requirements():
        return False
    
    if not check_env_file():
        return False
    
    # Set environment variables
    os.environ["PYTHONPATH"] = str(Path.cwd())
    
    # Run the application (using simple version first)
    cmd = [
        sys.executable, "-m", "uvicorn",
        "src.api.simple_main:app",
        "--reload",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--log-level", "info"
    ]
    
    print("🌐 Starting server at http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")
    print("⏹️  Press Ctrl+C to stop")
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting server: {e}")
        return False
    
    return True


def run_production():
    """Run in production mode"""
    print("🚀 Starting Visionary Prompt Architect in production mode...")
    
    if not check_requirements():
        return False
    
    if not check_env_file():
        return False
    
    # Set environment variables
    os.environ["PYTHONPATH"] = str(Path.cwd())
    os.environ["ENVIRONMENT"] = "production"
    
    # Run with gunicorn for production
    cmd = [
        sys.executable, "-m", "gunicorn",
        "src.api.main:app",
        "-w", "4",
        "-k", "uvicorn.workers.UvicornWorker",
        "--bind", "0.0.0.0:8000",
        "--access-logfile", "-",
        "--error-logfile", "-"
    ]
    
    print("🌐 Starting production server at http://localhost:8000")
    print("⏹️  Press Ctrl+C to stop")
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting server: {e}")
        return False
    
    return True


def run_tests():
    """Run the test suite"""
    print("🧪 Running test suite...")
    
    if not check_requirements():
        return False
    
    # Install test dependencies if needed
    try:
        import pytest
    except ImportError:
        print("📦 Installing test dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pytest", "pytest-asyncio", "pytest-cov"])
    
    # Run tests
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "--cov=src",
        "--cov-report=html",
        "--cov-report=term"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ All tests passed!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Tests failed: {e}")
        return False
    
    return True


def run_docker():
    """Run with Docker"""
    print("🐳 Starting with Docker...")
    
    # Check if Docker is available
    try:
        subprocess.run(["docker", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Docker not found. Please install Docker first.")
        return False
    
    # Build and run
    try:
        print("🔨 Building Docker image...")
        subprocess.run(["docker", "build", "-t", "prompt-architect:latest", "."], check=True)
        
        print("🚀 Starting container...")
        subprocess.run([
            "docker", "run", "-d",
            "--name", "prompt-architect",
            "-p", "8000:8000",
            "--env-file", ".env",
            "prompt-architect:latest"
        ], check=True)
        
        print("✅ Container started successfully!")
        print("🌐 Access at http://localhost:8000")
        print("📚 API Documentation: http://localhost:8000/docs")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Docker error: {e}")
        return False
    
    return True


def run_docker_compose():
    """Run with Docker Compose"""
    print("🐙 Starting with Docker Compose...")
    
    # Check if Docker Compose is available
    try:
        subprocess.run(["docker-compose", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Docker Compose not found. Please install Docker Compose first.")
        return False
    
    try:
        print("🚀 Starting all services...")
        subprocess.run(["docker-compose", "up", "-d"], check=True)
        
        print("✅ All services started!")
        print("🌐 API: http://localhost:8000")
        print("📚 API Docs: http://localhost:8000/docs")
        print("📊 Grafana: http://localhost:3000 (admin/admin)")
        print("📈 Prometheus: http://localhost:9090")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Docker Compose error: {e}")
        return False
    
    return True


def show_status():
    """Show system status"""
    print("📊 Visionary Prompt Architect Status")
    print("=" * 50)
    
    # Check if running
    try:
        import requests
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ API Server: Running")
            print(f"   Status: {data.get('status', 'unknown')}")
            print(f"   Version: {data.get('version', 'unknown')}")
            print(f"   Components: {len([k for k, v in data.get('components', {}).items() if v])} active")
        else:
            print("❌ API Server: Not responding")
    except Exception:
        print("❌ API Server: Not running")
    
    # Check Docker containers
    try:
        result = subprocess.run(["docker", "ps", "--filter", "name=prompt-architect", "--format", "table {{.Names}}\t{{.Status}}"], 
                              capture_output=True, text=True)
        if result.returncode == 0 and "prompt-architect" in result.stdout:
            print("✅ Docker: Container running")
        else:
            print("❌ Docker: No container running")
    except Exception:
        print("❌ Docker: Not available")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Visionary Prompt Architect - Run Script")
    parser.add_argument("command", nargs="?", default="dev", 
                       choices=["dev", "prod", "test", "docker", "compose", "status"],
                       help="Command to run (default: dev)")
    
    args = parser.parse_args()
    
    print("🎯 Visionary Prompt Architect")
    print("Author: Balaji Koneti")
    print("=" * 50)
    
    if args.command == "dev":
        success = run_development()
    elif args.command == "prod":
        success = run_production()
    elif args.command == "test":
        success = run_tests()
    elif args.command == "docker":
        success = run_docker()
    elif args.command == "compose":
        success = run_docker_compose()
    elif args.command == "status":
        show_status()
        return
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
