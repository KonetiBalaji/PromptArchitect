#!/usr/bin/env python3
"""
Basic Usage Example for Prompt Architect
Author: Balaji Koneti

Demonstrates basic usage of the enhanced Prompt Architect with
multi-LLM support, caching, and dynamic template selection.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_manager import LLMManager, LLMManagerConfig
from cache.cache_manager import CacheManager, CacheConfig
from prompt_builder import PromptBuilder, PromptRequest
from llm_providers.base import ProviderType, CompletionRequest


async def main():
    """Main function demonstrating basic usage"""
    print("🚀 Prompt Architect - Basic Usage Example")
    print("=" * 50)
    
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
        
        # Initialize LLM manager
        print("\n🤖 Initializing LLM manager...")
        llm_config = LLMManagerConfig(
            default_provider=list(api_keys.keys())[0],
            fallback_providers=list(api_keys.keys())[1:],
            enable_caching=True,
            cache_config=cache_config
        )
        llm_manager = LLMManager(llm_config)
        await llm_manager.initialize(api_keys)
        print("✅ LLM manager initialized")
        
        # Initialize prompt builder
        print("\n🔧 Initializing prompt builder...")
        prompt_builder = PromptBuilder(llm_manager, cache_manager)
        print("✅ Prompt builder initialized")
        
        # Example 1: Basic prompt generation
        print("\n📝 Example 1: Basic Prompt Generation")
        print("-" * 40)
        
        request = PromptRequest(
            user_input="Summarize the key findings from this research paper about climate change",
            variables={
                "domain": "environmental science",
                "document_type": "research paper",
                "focus_areas": "key findings and implications",
                "topic": "climate change",
                "length": "5000",
                "sections": "introduction, methodology, results, discussion",
                "audience": "general public",
                "format": "markdown"
            }
        )
        
        generated_prompt = await prompt_builder.generate_prompt(request)
        print(f"Template used: {generated_prompt.template_id}")
        print(f"Generation time: {generated_prompt.generation_time:.2f}s")
        print(f"From cache: {generated_prompt.cached}")
        print("\nGenerated prompt:")
        print(generated_prompt.full_prompt)
        
        # Example 2: Get completion from LLM
        print("\n🤖 Example 2: Get LLM Completion")
        print("-" * 40)
        
        completion_request = CompletionRequest(
            model="gpt-4o-mini",  # Use mini model for cost efficiency
            temperature=0.7,
            max_tokens=500
        )
        
        response = await prompt_builder.generate_completion(request, completion_request)
        print(f"Provider: {response.provider}")
        print(f"Model: {response.model}")
        print(f"Response time: {response.response_time:.2f}s")
        print(f"Tokens used: {response.usage.get('total_tokens', 'N/A')}")
        print("\nResponse:")
        print(response.content)
        
        # Example 3: Test caching
        print("\n💾 Example 3: Testing Caching")
        print("-" * 40)
        
        # Generate the same prompt again
        cached_prompt = await prompt_builder.generate_prompt(request)
        print(f"Second generation time: {cached_prompt.generation_time:.2f}s")
        print(f"From cache: {cached_prompt.cached}")
        
        # Example 4: Different template
        print("\n📋 Example 4: Different Template")
        print("-" * 40)
        
        qa_request = PromptRequest(
            user_input="What are the main causes of climate change?",
            template_id="qa",
            variables={
                "domain": "environmental science",
                "subject_area": "climate science",
                "topic": "climate change causes",
                "question": "What are the main causes of climate change?",
                "context": "scientific understanding",
                "factors": "greenhouse gases, human activities, natural processes",
                "format": "structured list"
            }
        )
        
        qa_prompt = await prompt_builder.generate_prompt(qa_request)
        print(f"Template used: {qa_prompt.template_id}")
        print("\nGenerated prompt:")
        print(qa_prompt.full_prompt)
        
        # Example 5: Available models
        print("\n🔍 Example 5: Available Models")
        print("-" * 40)
        
        models = await llm_manager.get_available_models()
        for provider, model_list in models.items():
            print(f"{provider}: {', '.join(model_list)}")
        
        # Example 6: Provider status
        print("\n📊 Example 6: Provider Status")
        print("-" * 40)
        
        status = await llm_manager.get_provider_status()
        for provider, info in status.items():
            print(f"{provider}: {info['status']} (errors: {info['error_count']}, success: {info['success_count']})")
        
        # Example 7: Cache statistics
        print("\n📈 Example 7: Cache Statistics")
        print("-" * 40)
        
        cache_stats = await cache_manager.get_cache_stats()
        if cache_stats:
            print(f"Total keys: {cache_stats['cache_info']['total_keys']}")
            print(f"Hit rate: {cache_stats['metrics'].hit_rate:.1f}%")
            print(f"Total requests: {cache_stats['metrics'].total_requests}")
        
        print("\n✅ All examples completed successfully!")
        
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
