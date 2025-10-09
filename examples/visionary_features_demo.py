"""
Visionary Features Demo
Author: Balaji Koneti

Comprehensive demonstration of all visionary features including:
- Latest reasoning models (OpenAI o1/o3, Claude extended thinking, Gemini 2.0)
- Advanced chain-of-thought reasoning
- AI-powered template generation
- Dynamic template selection
- Multi-provider orchestration
"""

import asyncio
import time
from typing import Dict, Any

from src.llm_manager import LLMManager
from src.reasoning_manager import ReasoningManager, ReasoningConfig, ReasoningStrategy
from src.chain_of_thought.sequential_chain import SequentialChain, SequentialChainConfig
from src.chain_of_thought.parallel_chain import ParallelChain, ParallelChainConfig
from src.chain_of_thought.tree_reasoning import TreeReasoning, TreeReasoningConfig
from src.chain_of_thought.reflection import Reflection, ReflectionConfig
from src.templates.template_generator import TemplateGenerator, TemplateGenerationRequest, TemplateCategory, TemplateComplexity
from src.templates.template_registry import template_registry
from src.templates.dynamic_selector import DynamicSelector
from src.llm_providers.base import CompletionRequest, ProviderType


class VisionaryFeaturesDemo:
    """
    Comprehensive demo of all visionary features
    """
    
    def __init__(self):
        """Initialize the demo with all components"""
        self.llm_manager = None
        self.reasoning_manager = None
        self.template_generator = None
        self.dynamic_selector = None
        self.chain_implementations = {}
        
    async def initialize(self):
        """Initialize all components"""
        print("🚀 Initializing Visionary Prompt Architect...")
        
        # Initialize LLM Manager
        self.llm_manager = LLMManager()
        await self.llm_manager.initialize()
        
        # Initialize Reasoning Manager
        self.reasoning_manager = ReasoningManager(self.llm_manager)
        
        # Initialize Template Generator
        self.template_generator = TemplateGenerator(self.llm_manager)
        
        # Initialize Dynamic Selector
        self.dynamic_selector = DynamicSelector(self.llm_manager)
        
        # Initialize Chain-of-Thought implementations
        self.chain_implementations = {
            "sequential": SequentialChain(self.llm_manager),
            "parallel": ParallelChain(self.llm_manager),
            "tree": TreeReasoning(self.llm_manager),
            "reflection": Reflection(self.llm_manager)
        }
        
        print("✅ All components initialized successfully!")
    
    async def demo_reasoning_models(self):
        """Demonstrate latest reasoning models"""
        print("\n🧠 Demo: Latest Reasoning Models")
        print("=" * 50)
        
        # Test OpenAI o1 models
        print("\n1. Testing OpenAI o1-preview (Reasoning Model)")
        try:
            request = CompletionRequest(
                prompt="Solve this complex math problem step by step: If a train leaves station A at 60 mph and another leaves station B at 80 mph, and they are 200 miles apart, when will they meet?",
                model="o1-preview",
                reasoning_effort="high",
                max_completion_tokens=2000
            )
            
            response = await self.llm_manager.generate_completion(request, preferred_provider=ProviderType.OPENAI)
            print(f"✅ OpenAI o1 Response: {response.content[:200]}...")
            print(f"   Confidence: {response.confidence_score}")
            print(f"   Reasoning Steps: {len(response.reasoning_steps) if response.reasoning_steps else 'N/A'}")
            
        except Exception as e:
            print(f"❌ OpenAI o1 Error: {e}")
        
        # Test Claude extended thinking
        print("\n2. Testing Claude 3.7 Sonnet (Extended Thinking)")
        try:
            request = CompletionRequest(
                prompt="Analyze the ethical implications of artificial intelligence in healthcare decision-making.",
                model="claude-3-7-sonnet-20241218",
                extended_thinking=True,
                max_tokens=1500
            )
            
            response = await self.llm_manager.generate_completion(request, preferred_provider=ProviderType.CLAUDE)
            print(f"✅ Claude Extended Thinking Response: {response.content[:200]}...")
            print(f"   Thinking Blocks: {len(response.thinking_blocks) if response.thinking_blocks else 'N/A'}")
            print(f"   Confidence: {response.confidence_score}")
            
        except Exception as e:
            print(f"❌ Claude Extended Thinking Error: {e}")
        
        # Test Gemini 2.0 reasoning
        print("\n3. Testing Gemini 2.0 Flash Thinking (Reasoning + Grounding)")
        try:
            request = CompletionRequest(
                prompt="What are the latest developments in quantum computing?",
                model="gemini-2.0-flash-thinking-exp",
                thinking_mode=True,
                grounding=True,
                search_web=True,
                max_tokens=1000
            )
            
            response = await self.llm_manager.generate_completion(request, preferred_provider=ProviderType.GEMINI)
            print(f"✅ Gemini 2.0 Response: {response.content[:200]}...")
            print(f"   Grounded Sources: {len(response.grounded_sources) if response.grounded_sources else 'N/A'}")
            print(f"   Confidence: {response.confidence_score}")
            
        except Exception as e:
            print(f"❌ Gemini 2.0 Error: {e}")
    
    async def demo_chain_of_thought(self):
        """Demonstrate advanced chain-of-thought reasoning"""
        print("\n🔗 Demo: Advanced Chain-of-Thought Reasoning")
        print("=" * 50)
        
        # Test Sequential Chain
        print("\n1. Sequential Chain-of-Thought")
        try:
            config = SequentialChainConfig(
                max_steps=5,
                verification_enabled=True,
                confidence_threshold=0.7
            )
            
            request = CompletionRequest(
                prompt="Design a sustainable city for 1 million people",
                model="gpt-4o",
                max_tokens=2000
            )
            
            result = await self.chain_implementations["sequential"].reason(request, config)
            print(f"✅ Sequential Chain Result:")
            print(f"   Steps: {len(result.steps)}")
            print(f"   Confidence: {result.overall_confidence:.2f}")
            print(f"   Time: {result.total_time:.2f}s")
            print(f"   Final Answer: {result.final_answer[:150]}...")
            
        except Exception as e:
            print(f"❌ Sequential Chain Error: {e}")
        
        # Test Parallel Chain
        print("\n2. Parallel Chain-of-Thought")
        try:
            config = ParallelChainConfig(
                num_paths=3,
                consensus_threshold=0.7,
                conflict_resolution_enabled=True
            )
            
            request = CompletionRequest(
                prompt="Evaluate the pros and cons of remote work",
                model="gpt-4o",
                max_tokens=1500
            )
            
            result = await self.chain_implementations["parallel"].reason(request, config)
            print(f"✅ Parallel Chain Result:")
            print(f"   Paths: {len(result.paths)}")
            print(f"   Consensus Level: {result.consensus.agreement_level:.2f}")
            print(f"   Time: {result.total_time:.2f}s")
            print(f"   Final Answer: {result.final_answer[:150]}...")
            
        except Exception as e:
            print(f"❌ Parallel Chain Error: {e}")
        
        # Test Tree-Based Reasoning
        print("\n3. Tree-Based Reasoning")
        try:
            config = TreeReasoningConfig(
                max_depth=3,
                max_branches_per_node=3,
                pruning_strategy="adaptive"
            )
            
            request = CompletionRequest(
                prompt="Develop a comprehensive cybersecurity strategy",
                model="gpt-4o",
                max_tokens=2000
            )
            
            result = await self.chain_implementations["tree"].reason(request, config)
            print(f"✅ Tree-Based Result:")
            print(f"   Nodes Explored: {result.nodes_explored}")
            print(f"   Nodes Pruned: {result.nodes_pruned}")
            print(f"   Best Path Length: {len(result.best_path)}")
            print(f"   Time: {result.total_time:.2f}s")
            print(f"   Final Answer: {result.final_answer[:150]}...")
            
        except Exception as e:
            print(f"❌ Tree-Based Error: {e}")
        
        # Test Reflection
        print("\n4. Reflective Reasoning")
        try:
            config = ReflectionConfig(
                max_reflection_steps=3,
                confidence_threshold=0.8,
                enable_error_detection=True
            )
            
            request = CompletionRequest(
                prompt="Create a business plan for a sustainable energy startup",
                model="gpt-4o",
                max_tokens=2000
            )
            
            result = await self.chain_implementations["reflection"].reflect_and_refine(request, config)
            print(f"✅ Reflection Result:")
            print(f"   Reflection Steps: {len(result.reflection_steps)}")
            print(f"   Confidence Improvement: {result.confidence_improvement:.2f}")
            print(f"   Time: {result.total_time:.2f}s")
            print(f"   Final Answer: {result.final_answer[:150]}...")
            
        except Exception as e:
            print(f"❌ Reflection Error: {e}")
    
    async def demo_ai_templates(self):
        """Demonstrate AI-powered template generation"""
        print("\n🎨 Demo: AI-Powered Template Generation")
        print("=" * 50)
        
        # Generate a research template
        print("\n1. Generating Research Template")
        try:
            request = TemplateGenerationRequest(
                category=TemplateCategory.RESEARCH,
                complexity=TemplateComplexity.EXPERT,
                use_case="Comprehensive market research analysis for new product launch",
                requirements=[
                    "Include competitive analysis",
                    "Market sizing and segmentation",
                    "Customer persona development",
                    "SWOT analysis"
                ],
                target_audience="business analysts",
                reasoning_enabled=True
            )
            
            template = await self.template_generator.generate_template(request)
            print(f"✅ Generated Template:")
            print(f"   ID: {template.template_id}")
            print(f"   Name: {template.name}")
            print(f"   Variables: {len(template.variables)}")
            print(f"   Reasoning Enabled: {template.reasoning_integration}")
            print(f"   Content Preview: {template.template_content[:200]}...")
            
        except Exception as e:
            print(f"❌ Template Generation Error: {e}")
        
        # Generate a creative template
        print("\n2. Generating Creative Template")
        try:
            request = TemplateGenerationRequest(
                category=TemplateCategory.CREATIVE,
                complexity=TemplateComplexity.COMPLEX,
                use_case="Brand storytelling for sustainable fashion brand",
                requirements=[
                    "Emotional connection with audience",
                    "Sustainability messaging",
                    "Brand values integration",
                    "Call-to-action optimization"
                ],
                target_audience="marketing professionals",
                reasoning_enabled=True
            )
            
            template = await self.template_generator.generate_template(request)
            print(f"✅ Generated Template:")
            print(f"   ID: {template.template_id}")
            print(f"   Name: {template.name}")
            print(f"   Variables: {len(template.variables)}")
            print(f"   Content Preview: {template.template_content[:200]}...")
            
        except Exception as e:
            print(f"❌ Creative Template Error: {e}")
        
        # Show template registry statistics
        print("\n3. Template Registry Statistics")
        try:
            analytics = template_registry.get_template_analytics()
            print(f"✅ Template Registry:")
            print(f"   Total Templates: {analytics['total_templates']}")
            print(f"   Categories: {len(analytics['categories'])}")
            print(f"   Average Quality: {analytics['average_quality_score']:.2f}")
            print(f"   Categories: {', '.join(analytics['categories'])}")
            
        except Exception as e:
            print(f"❌ Template Registry Error: {e}")
    
    async def demo_dynamic_selection(self):
        """Demonstrate dynamic template selection"""
        print("\n🎯 Demo: Dynamic Template Selection")
        print("=" * 50)
        
        # Test dynamic selection with different inputs
        test_cases = [
            {
                "input": "I need to analyze the financial performance of a tech startup for potential investment",
                "expected_category": "financial"
            },
            {
                "input": "Help me write a legal contract for software licensing agreement",
                "expected_category": "legal"
            },
            {
                "input": "Create a comprehensive research proposal for studying climate change impacts",
                "expected_category": "research"
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{i}. Testing: {test_case['input'][:50]}...")
            try:
                # Get available templates
                available_templates = template_registry.get_all_templates()
                
                # Use dynamic selector
                result = await self.dynamic_selector.select_template(
                    test_case["input"],
                    available_templates
                )
                
                print(f"✅ Selected Template:")
                print(f"   Template ID: {result.template_id}")
                print(f"   Confidence: {result.confidence:.2f}")
                print(f"   Intent: {result.detected_intent}")
                print(f"   Reasoning: {result.reasoning[:100]}...")
                
            except Exception as e:
                print(f"❌ Dynamic Selection Error: {e}")
    
    async def demo_reasoning_orchestration(self):
        """Demonstrate reasoning orchestration across providers"""
        print("\n🎼 Demo: Reasoning Orchestration")
        print("=" * 50)
        
        # Test unified reasoning interface
        print("\n1. Unified Reasoning Interface")
        try:
            config = ReasoningConfig(
                strategy=ReasoningStrategy.HYBRID,
                max_steps=4,
                verification_enabled=True,
                confidence_threshold=0.8
            )
            
            request = CompletionRequest(
                prompt="Develop a comprehensive AI ethics framework for healthcare applications",
                model="gpt-4o",
                max_tokens=2000
            )
            
            result = await self.reasoning_manager.reason(request, config)
            print(f"✅ Reasoning Orchestration Result:")
            print(f"   Strategy: {result.strategy}")
            print(f"   Steps: {len(result.steps)}")
            print(f"   Confidence: {result.confidence:.2f}")
            print(f"   Provider: {result.provider_used}")
            print(f"   Time: {result.total_time:.2f}s")
            print(f"   Final Answer: {result.final_answer[:150]}...")
            
        except Exception as e:
            print(f"❌ Reasoning Orchestration Error: {e}")
    
    async def demo_performance_metrics(self):
        """Demonstrate performance metrics and analytics"""
        print("\n📊 Demo: Performance Metrics & Analytics")
        print("=" * 50)
        
        # Show reasoning statistics
        print("\n1. Reasoning Statistics")
        try:
            stats = self.reasoning_manager.get_reasoning_stats()
            print(f"✅ Reasoning Manager Stats:")
            print(f"   Total Sessions: {stats['total_reasoning_sessions']}")
            print(f"   Average Confidence: {stats['average_confidence']:.2f}")
            print(f"   Average Time: {stats['average_time']:.2f}s")
            print(f"   Verification Pass Rate: {stats['verification_pass_rate']:.2f}")
            
        except Exception as e:
            print(f"❌ Reasoning Stats Error: {e}")
        
        # Show template generation statistics
        print("\n2. Template Generation Statistics")
        try:
            stats = self.template_generator.get_generation_stats()
            print(f"✅ Template Generator Stats:")
            print(f"   Total Templates: {stats['total_templates_generated']}")
            print(f"   Categories: {list(stats['category_distribution'].keys())}")
            print(f"   Reasoning Enabled: {stats['reasoning_enabled_templates']}")
            print(f"   Avg Variables: {stats['average_variables_per_template']:.1f}")
            
        except Exception as e:
            print(f"❌ Template Stats Error: {e}")
        
        # Show provider status
        print("\n3. Provider Status")
        try:
            for provider_type, provider_info in self.llm_manager.providers.items():
                print(f"✅ {provider_type.value.title()}:")
                print(f"   Status: {provider_info.status.value}")
                print(f"   Models: {len(provider_info.available_models)}")
                print(f"   Response Time: {provider_info.avg_response_time:.2f}s")
                print(f"   Success Rate: {provider_info.success_rate:.2f}")
                
        except Exception as e:
            print(f"❌ Provider Status Error: {e}")
    
    async def run_comprehensive_demo(self):
        """Run the complete visionary features demo"""
        print("🌟 Visionary Prompt Architect - Comprehensive Demo")
        print("=" * 60)
        print("Author: Balaji Koneti")
        print("Features: Latest Models + Chain-of-Thought + AI Templates")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            # Initialize all components
            await self.initialize()
            
            # Run all demos
            await self.demo_reasoning_models()
            await self.demo_chain_of_thought()
            await self.demo_ai_templates()
            await self.demo_dynamic_selection()
            await self.demo_reasoning_orchestration()
            await self.demo_performance_metrics()
            
            total_time = time.time() - start_time
            
            print("\n🎉 Demo Complete!")
            print("=" * 50)
            print(f"Total Demo Time: {total_time:.2f} seconds")
            print("All visionary features demonstrated successfully!")
            print("Ready for production deployment! 🚀")
            
        except Exception as e:
            print(f"\n❌ Demo Error: {e}")
            raise


async def main():
    """Main demo function"""
    demo = VisionaryFeaturesDemo()
    await demo.run_comprehensive_demo()


if __name__ == "__main__":
    asyncio.run(main())
