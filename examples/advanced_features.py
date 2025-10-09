#!/usr/bin/env python3
"""
Advanced Features Example for Prompt Architect
Author: Balaji Koneti

Demonstrates advanced features including dynamic template selection,
multi-provider orchestration, caching, and performance monitoring.
"""

import asyncio
import os
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_manager import LLMManager, LLMManagerConfig, LoadBalancingStrategy
from cache.cache_manager import CacheManager, CacheConfig
from prompt_builder import PromptBuilder, PromptRequest
from templates.dynamic_selector import DynamicSelector
from llm_providers.base import ProviderType, CompletionRequest


async def main():
    """Main function demonstrating advanced features"""
    print("🚀 Prompt Architect - Advanced Features Example")
    print("=" * 60)
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Check for required API keys
    api_keys = {
        ProviderType.OPENAI: os.getenv("OPENAI_API_KEY"),
        ProviderType.CLAUDE: os.getenv("ANTHROPIC_API_KEY"),
        ProviderType.GEMINI: os.getenv("GOOGLE_API_KEY")
    }
    
    # Filter out None values
    api_keys = {k: v for k, v in api_keys.items() if v}
    
    if not api_keys:
        print("❌ No API keys found. Please set at least one of:")
        print("   - OPENAI_API_KEY")
        print("   - ANTHROPIC_API_KEY") 
        print("   - GOOGLE_API_KEY")
        return
    
    print(f"✅ Found API keys for: {', '.join(api_keys.keys())}")
    
    try:
        # Initialize cache manager
        print("\n📦 Initializing cache manager...")
        cache_config = CacheConfig(
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
            default_ttl=3600,
            prompt_ttl=7200,
            response_ttl=1800
        )
        cache_manager = CacheManager(cache_config)
        await cache_manager.connect()
        print("✅ Cache manager initialized")
        
        # Initialize LLM manager with advanced configuration
        print("\n🤖 Initializing LLM manager with advanced features...")
        llm_config = LLMManagerConfig(
            default_provider=list(api_keys.keys())[0],
            fallback_providers=list(api_keys.keys())[1:],
            load_balancing_strategy=LoadBalancingStrategy.LEAST_COST,
            enable_caching=True,
            cache_config=cache_config,
            max_retries=3,
            retry_delay=1.0,
            cost_threshold=0.5
        )
        llm_manager = LLMManager(llm_config)
        await llm_manager.initialize(api_keys)
        print("✅ LLM manager initialized with advanced features")
        
        # Initialize prompt builder
        print("\n🔧 Initializing prompt builder...")
        prompt_builder = PromptBuilder(llm_manager, cache_manager)
        print("✅ Prompt builder initialized")
        
        # Initialize dynamic selector
        print("\n🧠 Initializing dynamic selector...")
        dynamic_selector = DynamicSelector(llm_manager)
        print("✅ Dynamic selector initialized")
        
        # Example 1: Dynamic template selection
        print("\n🎯 Example 1: Dynamic Template Selection")
        print("-" * 50)
        
        test_inputs = [
            "Write a creative story about a robot learning to paint",
            "Analyze the sales data and identify trends",
            "What is the capital of France?",
            "Generate Python code for a REST API",
            "Summarize this research paper on machine learning"
        ]
        
        for i, user_input in enumerate(test_inputs, 1):
            print(f"\nTest {i}: {user_input}")
            
            # Get available templates
            available_templates = prompt_builder.list_templates()
            
            # Use dynamic selector
            classification = await dynamic_selector.select_template(
                user_input, available_templates
            )
            
            print(f"  Selected template: {classification.template_id}")
            print(f"  Confidence: {classification.confidence:.2f}")
            print(f"  Intent: {classification.detected_intent}")
            print(f"  Domain: {classification.detected_domain}")
            print(f"  Reasoning: {classification.reasoning}")
            
            if classification.alternative_templates:
                print(f"  Alternatives: {[alt[0] for alt in classification.alternative_templates[:2]]}")
        
        # Example 2: Multi-provider comparison
        print("\n🔄 Example 2: Multi-Provider Comparison")
        print("-" * 50)
        
        test_prompt = "Explain quantum computing in simple terms"
        
        for provider_type in api_keys.keys():
            print(f"\nTesting {provider_type.value}...")
            
            try:
                start_time = time.time()
                
                request = PromptRequest(
                    user_input=test_prompt,
                    provider=provider_type,
                    variables={
                        "domain": "technology",
                        "audience": "general public",
                        "format": "text"
                    }
                )
                
                response = await prompt_builder.generate_completion(
                    request,
                    CompletionRequest(
                        model="gpt-4o-mini" if provider_type == ProviderType.OPENAI else None,
                        temperature=0.7,
                        max_tokens=300
                    )
                )
                
                end_time = time.time()
                
                print(f"  Provider: {response.provider}")
                print(f"  Model: {response.model}")
                print(f"  Response time: {response.response_time:.2f}s")
                print(f"  Tokens: {response.usage.get('total_tokens', 'N/A')}")
                print(f"  Content preview: {response.content[:100]}...")
                
            except Exception as e:
                print(f"  Error: {str(e)}")
        
        # Example 3: Load balancing strategies
        print("\n⚖️ Example 3: Load Balancing Strategies")
        print("-" * 50)
        
        strategies = [
            LoadBalancingStrategy.PRIORITY,
            LoadBalancingStrategy.LEAST_COST,
            LoadBalancingStrategy.FASTEST,
            LoadBalancingStrategy.ROUND_ROBIN
        ]
        
        for strategy in strategies:
            print(f"\nTesting {strategy.value} strategy...")
            
            # Update LLM manager configuration
            llm_manager.config.load_balancing_strategy = strategy
            
            # Test multiple requests
            start_time = time.time()
            responses = []
            
            for i in range(3):
                request = PromptRequest(
                    user_input=f"Test request {i+1}: What is machine learning?",
                    variables={"domain": "technology", "audience": "general"}
                )
                
                response = await prompt_builder.generate_completion(
                    request,
                    CompletionRequest(
                        model="gpt-4o-mini",
                        temperature=0.7,
                        max_tokens=200
                    )
                )
                responses.append(response)
            
            end_time = time.time()
            
            # Analyze results
            providers_used = [r.provider for r in responses]
            avg_response_time = sum(r.response_time for r in responses) / len(responses)
            
            print(f"  Providers used: {providers_used}")
            print(f"  Average response time: {avg_response_time:.2f}s")
            print(f"  Total time: {end_time - start_time:.2f}s")
        
        # Example 4: Cache performance analysis
        print("\n💾 Example 4: Cache Performance Analysis")
        print("-" * 50)
        
        # Clear cache first
        await cache_manager.clear_cache()
        
        # Test cache performance
        test_requests = [
            "What is artificial intelligence?",
            "Explain blockchain technology",
            "What is artificial intelligence?",  # Duplicate for cache test
            "How does machine learning work?",
            "Explain blockchain technology"  # Another duplicate
        ]
        
        cache_hits = 0
        total_time = 0
        
        for i, user_input in enumerate(test_requests, 1):
            print(f"\nRequest {i}: {user_input}")
            
            start_time = time.time()
            
            request = PromptRequest(
                user_input=user_input,
                variables={"domain": "technology", "audience": "general"}
            )
            
            response = await prompt_builder.generate_completion(
                request,
                CompletionRequest(
                    model="gpt-4o-mini",
                    temperature=0.7,
                    max_tokens=200
                )
            )
            
            end_time = time.time()
            request_time = end_time - start_time
            total_time += request_time
            
            # Check if this was a cache hit (simplified check)
            if request_time < 0.1:  # Very fast response likely from cache
                cache_hits += 1
                print(f"  ⚡ Cache hit! Time: {request_time:.3f}s")
            else:
                print(f"  🔄 Cache miss. Time: {request_time:.2f}s")
        
        print(f"\nCache Performance Summary:")
        print(f"  Total requests: {len(test_requests)}")
        print(f"  Cache hits: {cache_hits}")
        print(f"  Hit rate: {cache_hits/len(test_requests)*100:.1f}%")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average time: {total_time/len(test_requests):.2f}s")
        
        # Example 5: Performance monitoring
        print("\n📊 Example 5: Performance Monitoring")
        print("-" * 50)
        
        # Get provider status
        status = await llm_manager.get_provider_status()
        print("Provider Status:")
        for provider, info in status.items():
            print(f"  {provider}:")
            print(f"    Status: {info['status']}")
            print(f"    Success count: {info['success_count']}")
            print(f"    Error count: {info['error_count']}")
            print(f"    Total cost: ${info['total_cost']:.4f}")
            print(f"    Avg response time: {info['avg_response_time']:.2f}s")
        
        # Get cache statistics
        cache_stats = await cache_manager.get_cache_stats()
        if cache_stats:
            print(f"\nCache Statistics:")
            print(f"  Total keys: {cache_stats['cache_info']['total_keys']}")
            print(f"  Hit rate: {cache_stats['metrics'].hit_rate:.1f}%")
            print(f"  Total requests: {cache_stats['metrics'].total_requests}")
            print(f"  Total savings: ${cache_stats['metrics'].total_savings:.4f}")
        
        # Get dynamic selector stats
        selector_stats = dynamic_selector.get_cache_stats()
        print(f"\nDynamic Selector Statistics:")
        print(f"  Cached classifications: {selector_stats['cached_classifications']}")
        print(f"  Templates with feedback: {selector_stats['templates_with_feedback']}")
        print(f"  Total feedback entries: {selector_stats['total_feedback_entries']}")
        
        # Example 6: Error handling and fallback
        print("\n🛡️ Example 6: Error Handling and Fallback")
        print("-" * 50)
        
        # Test with invalid model to trigger fallback
        try:
            request = PromptRequest(
                user_input="Test fallback mechanism",
                variables={"domain": "technology"}
            )
            
            # Use invalid model to test fallback
            response = await prompt_builder.generate_completion(
                request,
                CompletionRequest(
                    model="invalid-model-name",
                    temperature=0.7,
                    max_tokens=100
                )
            )
            
            print(f"✅ Fallback successful! Used provider: {response.provider}")
            
        except Exception as e:
            print(f"❌ Fallback failed: {str(e)}")
        
        print("\n✅ All advanced examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        if 'llm_manager' in locals():
            await llm_manager.close()
        if 'cache_manager' in locals():
            await cache_manager.disconnect()
        print("\n🧹 Cleanup completed")


if __name__ == "__main__":
    asyncio.run(main())
