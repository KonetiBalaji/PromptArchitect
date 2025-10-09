# Visionary Prompt Architect - Quick Start Guide
**Author: Balaji Koneti**

## 🚀 Quick Start (5 Minutes)

### Prerequisites
- Python 3.11+
- Git
- API Keys (OpenAI, Anthropic, Google)

### Step 1: Clone and Setup
```bash
# Clone the repository
git clone <your-repo-url>
cd prompt-architect

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure Environment
```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your API keys
# Required keys:
OPENAI_API_KEY=your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here
GOOGLE_API_KEY=your-google-api-key-here
SECRET_KEY=your-secret-key-here
```

### Step 3: Run the Application
```bash
# Start the API server
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Step 4: Test the System
```bash
# Test health endpoint
curl http://localhost:8000/health

# Test completion endpoint (requires authentication)
curl -X POST "http://localhost:8000/completion" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "prompt": "What is 2+2?",
    "model": "gpt-4o-mini",
    "temperature": 0.0
  }'
```

### Step 5: Access Documentation
- **API Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## 🐳 Docker Quick Start

### Step 1: Build and Run
```bash
# Build the Docker image
docker build -t prompt-architect:latest .

# Run with environment variables
docker run -d \
  --name prompt-architect \
  -p 8000:8000 \
  -e OPENAI_API_KEY=your-key \
  -e ANTHROPIC_API_KEY=your-key \
  -e GOOGLE_API_KEY=your-key \
  -e SECRET_KEY=your-secret \
  prompt-architect:latest
```

### Step 2: Check Status
```bash
# Check if container is running
docker ps

# View logs
docker logs prompt-architect

# Test the API
curl http://localhost:8000/health
```

## 🐙 Docker Compose (Full Stack)

### Step 1: Setup Environment
```bash
# Create .env file
cat > .env << EOF
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
GOOGLE_API_KEY=your-google-api-key
SECRET_KEY=your-secret-key
POSTGRES_PASSWORD=password
REDIS_PASSWORD=password
EOF
```

### Step 2: Start All Services
```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f prompt-architect-api
```

### Step 3: Access Services
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090

## ☸️ Kubernetes Deployment

### Step 1: Setup Secrets
```bash
# Create namespace
kubectl apply -f k8s/namespace.yaml

# Create secrets (replace with your actual values)
kubectl create secret generic prompt-architect-secrets \
  --from-literal=openai-api-key=your-key \
  --from-literal=anthropic-api-key=your-key \
  --from-literal=google-api-key=your-key \
  --from-literal=secret-key=your-secret \
  --from-literal=database-url=postgresql://postgres:password@postgres:5432/prompt_architect \
  --from-literal=redis-url=redis://redis:6379/0 \
  -n prompt-architect
```

### Step 2: Deploy Application
```bash
# Deploy the application
kubectl apply -f k8s/deployment.yaml

# Check deployment status
kubectl get pods -n prompt-architect
kubectl get services -n prompt-architect
```

### Step 3: Access the Application
```bash
# Get service URL
kubectl get service prompt-architect-api-loadbalancer -n prompt-architect

# Port forward for local access
kubectl port-forward service/prompt-architect-api-service 8000:80 -n prompt-architect
```

## 🧪 Testing the System

### Basic API Test
```bash
# Test health endpoint
curl http://localhost:8000/health

# Expected response:
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00",
  "version": "1.0.0",
  "components": {
    "llm_manager": true,
    "reasoning_manager": true,
    "template_generator": true,
    "auth_manager": true,
    "rate_limiter": true
  }
}
```

### Python Script Test
```python
import asyncio
import httpx

async def test_api():
    async with httpx.AsyncClient() as client:
        # Test health
        response = await client.get("http://localhost:8000/health")
        print(f"Health: {response.json()}")
        
        # Test completion (requires auth token)
        headers = {"Authorization": "Bearer YOUR_TOKEN"}
        data = {
            "prompt": "What is the capital of France?",
            "model": "gpt-4o-mini",
            "temperature": 0.0
        }
        response = await client.post(
            "http://localhost:8000/completion",
            json=data,
            headers=headers
        )
        print(f"Completion: {response.json()}")

asyncio.run(test_api())
```

## 🔧 Configuration Options

### Environment Variables
```bash
# Required
OPENAI_API_KEY=your-key
ANTHROPIC_API_KEY=your-key
GOOGLE_API_KEY=your-key
SECRET_KEY=your-secret

# Optional
DATABASE_URL=postgresql://user:pass@localhost:5432/db
REDIS_URL=redis://localhost:6379/0
DEBUG=true
LOG_LEVEL=INFO
```

### Configuration File
Edit `config/config.yaml` to customize:
- Model preferences
- Reasoning strategies
- Template settings
- Monitoring options

## 🐛 Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Make sure you're in the project directory
cd prompt-architect

# Check Python path
export PYTHONPATH=$PWD:$PYTHONPATH
```

#### 2. API Key Issues
```bash
# Verify API keys are set
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY
echo $GOOGLE_API_KEY
```

#### 3. Port Already in Use
```bash
# Find process using port 8000
lsof -i :8000

# Kill the process
kill -9 <PID>

# Or use different port
python -m uvicorn src.api.main:app --port 8001
```

#### 4. Database Connection Issues
```bash
# Check if PostgreSQL is running
pg_isready

# Test connection
psql -h localhost -U postgres -d prompt_architect
```

### Debug Mode
```bash
# Enable debug logging
export DEBUG=true
export LOG_LEVEL=DEBUG

# Run with debug
python -m uvicorn src.api.main:app --reload --log-level debug
```

## 📊 Monitoring

### Health Checks
```bash
# Application health
curl http://localhost:8000/health

# Database health
curl http://localhost:8000/health/database

# LLM providers health
curl http://localhost:8000/health/providers
```

### Metrics
```bash
# Prometheus metrics
curl http://localhost:8000/metrics

# Grafana dashboard
open http://localhost:3000
```

## 🚀 Production Deployment

### Using Docker Compose
```bash
# Production deployment
docker-compose -f docker-compose.prod.yml up -d
```

### Using Kubernetes
```bash
# Production deployment
kubectl apply -f k8s/production/
```

## 📚 Next Steps

1. **Explore the API**: Visit http://localhost:8000/docs
2. **Run Examples**: Check `examples/` directory
3. **Read Documentation**: See `DOCUMENTATION.md`
4. **Run Tests**: `pytest tests/ -v`
5. **Monitor Performance**: Access Grafana dashboard

## 🆘 Getting Help

- **Documentation**: `DOCUMENTATION.md`
- **API Docs**: http://localhost:8000/docs
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions

---

**Happy Prompting! 🎉**
