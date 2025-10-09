# 🚀 Prompt Architect Setup Guide

**Author: Balaji Koneti**

Complete setup guide for the enhanced Prompt Architect with multi-LLM support, caching, and dynamic template selection.

## 📋 Prerequisites

- Python 3.8 or higher
- Redis server (for caching)
- API keys for at least one LLM provider:
  - OpenAI API key
  - Anthropic Claude API key
  - Google Gemini API key

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/prompt-architect.git
cd prompt-architect
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Redis (Required for Caching)

#### Option A: Local Redis Installation

**Windows:**
```bash
# Download Redis from https://github.com/microsoftarchive/redis/releases
# Or use Chocolatey:
choco install redis-64
```

**macOS:**
```bash
# Using Homebrew:
brew install redis
brew services start redis
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install redis-server
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

#### Option B: Docker Redis

```bash
docker run -d --name redis -p 6379:6379 redis:alpine
```

#### Option C: Redis Cloud (Free Tier)

1. Sign up at [Redis Cloud](https://redis.com/try-free/)
2. Create a free database
3. Get connection details

### 5. Configure Environment Variables

```bash
# Copy the example environment file
cp env.example .env

# Edit .env with your API keys
nano .env  # or use your preferred editor
```

**Required Environment Variables:**

```env
# At least one LLM provider API key is required
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GOOGLE_API_KEY=your_google_api_key_here

# Redis configuration
REDIS_URL=redis://localhost:6379

# Optional: Custom Redis configuration
REDIS_PASSWORD=your_redis_password
REDIS_DB=0
```

### 6. Verify Installation

```bash
# Run the simple test
python examples/simple_test.py
```

If successful, you should see:
```
✅ OpenAI API key found
✅ Cache manager connected
✅ LLM manager initialized
✅ Prompt builder initialized
🎉 All tests passed successfully!
```

## 🔧 Configuration

### Basic Configuration

The system uses `config/config.yaml` for configuration. Key settings:

```yaml
# LLM Provider Configuration
llm_providers:
  default_provider: "openai"  # openai, claude, gemini
  fallback_providers:
    - "claude"
    - "gemini"
  load_balancing_strategy: "priority"  # priority, least_cost, fastest, round_robin

# Cache Configuration
cache:
  enabled: true
  redis_url: "redis://localhost:6379"
  default_ttl: 3600  # 1 hour
  prompt_ttl: 7200   # 2 hours
  response_ttl: 1800 # 30 minutes
```

### Advanced Configuration

#### Load Balancing Strategies

- **`priority`**: Use providers in priority order
- **`least_cost`**: Choose the cheapest provider
- **`fastest`**: Choose the fastest provider
- **`round_robin`**: Distribute requests evenly

#### Cache TTL Settings

- **`default_ttl`**: Default cache time-to-live
- **`prompt_ttl`**: How long to cache generated prompts
- **`response_ttl`**: How long to cache LLM responses
- **`template_ttl`**: How long to cache template compilations

## 🚀 Quick Start Examples

### 1. Basic Usage

```python
import asyncio
from src.llm_manager import LLMManager, LLMManagerConfig
from src.cache.cache_manager import CacheManager, CacheConfig
from src.prompt_builder import PromptBuilder, PromptRequest
from src.llm_providers.base import ProviderType, CompletionRequest

async def main():
    # Initialize components
    cache_config = CacheConfig()
    cache_manager = CacheManager(cache_config)
    await cache_manager.connect()
    
    llm_config = LLMManagerConfig(
        default_provider=ProviderType.OPENAI,
        enable_caching=True,
        cache_config=cache_config
    )
    
    llm_manager = LLMManager(llm_config)
    await llm_manager.initialize({
        ProviderType.OPENAI: "your-openai-key"
    })
    
    prompt_builder = PromptBuilder(llm_manager, cache_manager)
    
    # Generate prompt and get completion
    request = PromptRequest(
        user_input="Explain quantum computing",
        variables={"domain": "technology", "audience": "general"}
    )
    
    response = await prompt_builder.generate_completion(
        request,
        CompletionRequest(model="gpt-4o-mini", max_tokens=300)
    )
    
    print(f"Response: {response.content}")
    
    # Cleanup
    await llm_manager.close()
    await cache_manager.disconnect()

asyncio.run(main())
```

### 2. Multi-Provider Setup

```python
# Initialize with multiple providers
api_keys = {
    ProviderType.OPENAI: "your-openai-key",
    ProviderType.CLAUDE: "your-claude-key",
    ProviderType.GEMINI: "your-gemini-key"
}

llm_config = LLMManagerConfig(
    default_provider=ProviderType.OPENAI,
    fallback_providers=[ProviderType.CLAUDE, ProviderType.GEMINI],
    load_balancing_strategy=LoadBalancingStrategy.LEAST_COST
)

llm_manager = LLMManager(llm_config)
await llm_manager.initialize(api_keys)
```

### 3. Dynamic Template Selection

```python
from src.templates.dynamic_selector import DynamicSelector

# Initialize dynamic selector
dynamic_selector = DynamicSelector(llm_manager)

# Automatically select best template
classification = await dynamic_selector.select_template(
    "Write a creative story about AI",
    prompt_builder.list_templates()
)

print(f"Selected: {classification.template_id}")
print(f"Confidence: {classification.confidence}")
```

## 📊 Monitoring and Metrics

### Cache Performance

```python
# Get cache statistics
cache_stats = await cache_manager.get_cache_stats()
print(f"Hit rate: {cache_stats['metrics'].hit_rate:.1f}%")
print(f"Total requests: {cache_stats['metrics'].total_requests}")
print(f"Cost savings: ${cache_stats['metrics'].total_savings:.2f}")
```

### Provider Performance

```python
# Get provider status
status = await llm_manager.get_provider_status()
for provider, info in status.items():
    print(f"{provider}:")
    print(f"  Status: {info['status']}")
    print(f"  Success rate: {info['success_count']}")
    print(f"  Avg response time: {info['avg_response_time']:.2f}s")
    print(f"  Total cost: ${info['total_cost']:.4f}")
```

## 🧪 Testing

### Run All Tests

```bash
python -m pytest tests/ -v
```

### Run Specific Test Categories

```bash
# Test basic functionality
python -m pytest tests/test_basic.py -v

# Test with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

### Example Scripts

```bash
# Simple functionality test
python examples/simple_test.py

# Basic usage examples
python examples/basic_usage.py

# Advanced features demonstration
python examples/advanced_features.py
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Redis Connection Error

**Error:** `ConnectionError: Failed to connect to Redis`

**Solutions:**
- Ensure Redis is running: `redis-cli ping`
- Check Redis URL in `.env` file
- Verify Redis port (default: 6379)
- Check firewall settings

#### 2. API Key Authentication Error

**Error:** `AuthenticationError: OpenAI authentication failed`

**Solutions:**
- Verify API key is correct in `.env` file
- Check API key has sufficient credits
- Ensure API key has proper permissions

#### 3. Import Errors

**Error:** `ModuleNotFoundError: No module named 'src'`

**Solutions:**
- Ensure you're in the project root directory
- Check Python path includes the project directory
- Verify virtual environment is activated

#### 4. Provider Not Available

**Error:** `No available providers`

**Solutions:**
- Check at least one API key is set
- Verify provider initialization in logs
- Check network connectivity

### Debug Mode

Enable debug logging:

```python
import structlog
structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(20),  # DEBUG level
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=False,
)
```

### Performance Issues

#### Slow Response Times

1. **Check cache hit rate** - Low hit rates indicate caching issues
2. **Verify provider selection** - Ensure fastest provider is being used
3. **Check network latency** - Test API endpoints directly
4. **Monitor token usage** - Large prompts increase response time

#### High Costs

1. **Use cost-effective models** - GPT-4o-mini instead of GPT-4o
2. **Enable caching** - Reduces duplicate API calls
3. **Optimize prompts** - Shorter prompts = lower costs
4. **Monitor usage** - Set cost thresholds in configuration

## 🚀 Production Deployment

### Environment Setup

1. **Use production Redis** - Redis Cloud or managed Redis service
2. **Set secure API keys** - Use environment variables, not files
3. **Enable monitoring** - Set up logging and metrics collection
4. **Configure rate limiting** - Prevent API abuse

### Scaling Considerations

1. **Redis clustering** - For high-volume caching
2. **Load balancing** - Distribute requests across multiple instances
3. **Database persistence** - Store templates and metrics
4. **Monitoring** - Set up alerts for errors and performance

### Security Best Practices

1. **API key rotation** - Regularly rotate API keys
2. **Access control** - Implement user authentication
3. **Rate limiting** - Prevent abuse and control costs
4. **Audit logging** - Track all API usage

## 📚 Additional Resources

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Anthropic Claude API Documentation](https://docs.anthropic.com/)
- [Google Gemini API Documentation](https://ai.google.dev/docs)
- [Redis Documentation](https://redis.io/docs/)
- [Pydantic Documentation](https://docs.pydantic.dev/)

## 🤝 Support

For issues and questions:

1. Check this setup guide
2. Review the example scripts
3. Check the test suite for usage examples
4. Open an issue on GitHub

---

**Happy Prompt Engineering! 🚀**
