#!/usr/bin/env python3
"""
Simple Test Script for Prompt Architect
Author: Balaji Koneti

Quick test script to verify the installation and basic functionality.
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


async def simple_test():
    """Simple test function"""
    print("🧪 Prompt Architect - Simple Test")
    print("=" * 40)
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Check for API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("❌ OPENAI_API_KEY not found. Please set it in your .env file.")
        return False
    
    print("✅ OpenAI API key found")
    
    try:
        # Initialize components
        print("\n🔧 Initializing components...")
        
        # Cache manager (optional - will work without Redis)
        cache_config = CacheConfig(
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
            default_ttl=3600
        )
        cache_manager = CacheManager(cache_config)
        
        try:
            await cache_manager.connect()
            print("✅ Cache manager connected")
        except Exception as e:
            print(f"⚠️ Cache manager failed (continuing without cache): {str(e)}")
            cache_manager = None
        
        # LLM manager
        llm_config = LLMManagerConfig(
            default_provider=ProviderType.OPENAI,
            enable_caching=cache_manager is not None,
            cache_config=cache_config if cache_manager else None
        )
        llm_manager = LLMManager(llm_config)
        await llm_manager.initialize({ProviderType.OPENAI: openai_key})
        print("✅ LLM manager initialized")
        
        # Prompt builder
        prompt_builder = PromptBuilder(llm_manager, cache_manager)
        print("✅ Prompt builder initialized")
        
        # Test 1: Generate a simple prompt
        print("\n📝 Test 1: Generate Simple Prompt")
        print("-" * 30)
        
        request = PromptRequest(
            user_input="What is Python programming?",
            variables={
                "domain": "programming",
                "audience": "beginners",
                "format": "text"
            }
        )
        
        generated_prompt = await prompt_builder.generate_prompt(request)
        print(f"✅ Prompt generated using template: {generated_prompt.template_id}")
        print(f"Generation time: {generated_prompt.generation_time:.2f}s")
        
        # Test 2: Get LLM completion
        print("\n🤖 Test 2: Get LLM Completion")
        print("-" * 30)
        
        completion_request = CompletionRequest(
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=200
        )
        
        response = await prompt_builder.generate_completion(request, completion_request)
        print(f"✅ Completion received from {response.provider}")
        print(f"Response time: {response.response_time:.2f}s")
        print(f"Tokens used: {response.usage.get('total_tokens', 'N/A')}")
        print(f"Response preview: {response.content[:100]}...")
        
        # Test 3: List available templates
        print("\n📋 Test 3: List Available Templates")
        print("-" * 30)
        
        templates = prompt_builder.list_templates()
        print(f"✅ Found {len(templates)} templates:")
        for template in templates:
            print(f"  - {template.id}: {template.name}")
        
        # Test 4: Check available models
        print("\n🔍 Test 4: Check Available Models")
        print("-" * 30)
        
        models = await llm_manager.get_available_models()
        for provider, model_list in models.items():
            print(f"✅ {provider}: {', '.join(model_list)}")
        
        print("\n🎉 All tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        if 'llm_manager' in locals():
            await llm_manager.close()
        if 'cache_manager' in locals():
            try:
                await cache_manager.disconnect()
            except:
                pass


async def main():
    """Main function"""
    success = await simple_test()
    
    if success:
        print("\n✅ Simple test completed successfully!")
        print("You can now run the full examples:")
        print("  python examples/basic_usage.py")
        print("  python examples/advanced_features.py")
    else:
        print("\n❌ Simple test failed!")
        print("Please check your configuration and try again.")


if __name__ == "__main__":
    asyncio.run(main())
