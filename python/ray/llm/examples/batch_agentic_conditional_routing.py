"""
Conditional Routing Multi-Agent System

This example demonstrates conditional routing where:
1. Router Agent: Classifies user queries into categories
2. Specialized Agents: Handle queries based on their category

The workflow:
1. Router Agent receives a user query and classifies it
2. Query is routed to the appropriate specialized agent
3. Specialized agent processes and responds
"""

import ray
from ray import serve
from ray.serve.llm import LLMConfig, build_llm_deployment
from ray.serve.llm.openai_api_models import ChatCompletionRequest

from typing import Dict, Any, List

# Configure the LLM (same model for all agents, different system prompts)
llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="Qwen/Qwen3-4B-Instruct-2507-FP8",
        model_source="Qwen/Qwen3-4B-Instruct-2507-FP8",
    ),
    deployment_config=dict(
        autoscaling_config=dict(
            num_replicas=1,
        )
    ),
    engine_kwargs=dict(
        max_model_len=32768,
        trust_remote_code=True,
        gpu_memory_utilization=0.8,
    ),
)

# Set up deployment names
ROUTER_APP_NAME = "router_agent_app"
ROUTER_DEPLOYMENT_NAME = "RouterAgent"
WEATHER_APP_NAME = "weather_agent_app"
WEATHER_DEPLOYMENT_NAME = "WeatherAgent"
SCIENCE_APP_NAME = "science_agent_app"
SCIENCE_DEPLOYMENT_NAME = "ScienceAgent"
UNKNOWN_APP_NAME = "unknown_agent_app"
UNKNOWN_DEPLOYMENT_NAME = "UnknownAgent"

# System prompts for each agent
ROUTER_AGENT_SYSTEM_PROMPT = """You are a query router. Your role is to classify user queries into categories.

Categories:
- weather: Questions about weather conditions, forecasts, temperature, climate
- science: Questions about science, physics, chemistry, biology, mathematics, technology
- unknown: Queries that don't clearly fit weather or science categories

Classify each query into exactly one of these three categories: weather, science, or unknown."""

WEATHER_AGENT_SYSTEM_PROMPT = """You are a weather specialist. Your role is to provide accurate, helpful weather information.

When responding:
- Provide clear weather forecasts
- Include relevant details like temperature, conditions, and location
- Be concise and informative"""

SCIENCE_AGENT_SYSTEM_PROMPT = """You are a science specialist. Your role is to explain scientific concepts clearly and accurately.

When responding:
- Explain concepts in accessible language
- Provide accurate scientific information
- Use examples when helpful"""

UNKNOWN_AGENT_SYSTEM_PROMPT = """You are a helpful assistant. Your role is to handle queries that don't fit into specific categories.

When responding:
- Be polite and helpful
- Acknowledge that the query couldn't be categorized
- Suggest how the user might rephrase their query
- Offer general assistance"""


def sanitize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sanitize messages to ensure they're plain dicts."""
    sanitized = []
    for msg in messages:
        if isinstance(msg, dict):
            sanitized.append(msg)
        elif hasattr(msg, 'model_dump'):
            sanitized.append(msg.model_dump())
        elif hasattr(msg, 'dict'):
            sanitized.append(msg.dict())
        else:
            sanitized.append({"role": "user", "content": str(msg)})
    return sanitized


def llm_generate(
    handle, 
    messages: List[Dict[str, Any]], 
    system_prompt: str = None,
    structured_outputs: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Generate a response from an LLM deployment."""
    if system_prompt:
        full_messages = [{"role": "system", "content": system_prompt}] + messages
    else:
        full_messages = messages

    request = ChatCompletionRequest(
        model=llm_config.model_id,
        messages=sanitize_messages(full_messages),
        stream=False,
        structured_outputs=structured_outputs,
    )

    response_gen = handle.chat.remote(request)

    response = None
    for chunk in response_gen:
        response = chunk
    
    if response is None:
        raise RuntimeError("No response received from LLM deployment")

    if hasattr(response, 'model_dump'):
        return response.model_dump()
    elif hasattr(response, 'dict'):
        return response.dict()
    return response


def parse_routing_decision(router_response: Dict[str, Any]) -> str:
    """Parse routing decision from router agent response. Returns the category."""
    choices = router_response.get("choices", [])
    if not choices:
        raise ValueError("Router response has no choices")

    content = choices[0].get("message", {}).get("content", "").strip().lower()
    if not content:
        raise ValueError("Router response has no content")

    valid_categories = ["weather", "science", "unknown"]
    if content not in valid_categories:
        raise ValueError(f"Invalid category: '{content}'")

    return content


def process_query(query: str, row_id: int, handles: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single query through the conditional routing system."""
    router_handle = handles["router"]
    weather_handle = handles["weather"]
    science_handle = handles["science"]
    unknown_handle = handles["unknown"]

    # Step 1: Router classifies the query
    router_prompt = f"""Analyze the user query below and determine its category.

Categories:
- weather: For questions about weather conditions, forecasts, temperature, climate
- science: For questions about science, physics, chemistry, biology, mathematics, technology
- unknown: If the category is unclear or doesn't fit weather or science

Query: {query}

Classify this query into one of the categories: weather, science, or unknown."""

    structured_outputs = {"choice": ["weather", "science", "unknown"]}

    router_response = llm_generate(
        router_handle,
        [{"role": "user", "content": router_prompt}],
        system_prompt=ROUTER_AGENT_SYSTEM_PROMPT,
        structured_outputs=structured_outputs
    )

    category = parse_routing_decision(router_response)

    # Step 2: Route to appropriate specialized agent
    if category == "weather":
        handle = weather_handle
        system_prompt = WEATHER_AGENT_SYSTEM_PROMPT
    elif category == "science":
        handle = science_handle
        system_prompt = SCIENCE_AGENT_SYSTEM_PROMPT
    else:
        handle = unknown_handle
        system_prompt = UNKNOWN_AGENT_SYSTEM_PROMPT

    specialized_response = llm_generate(
        handle,
        [{"role": "user", "content": query}],
        system_prompt=system_prompt
    )

    # Extract final response
    specialized_choices = specialized_response.get("choices", [])
    if not specialized_choices:
        raise ValueError("Agent response has no choices")

    final_response = specialized_choices[0].get("message", {}).get("content", "")
    if not final_response:
        raise ValueError("Agent response has no content")

    return {
        "id": row_id,
        "query": query,
        "category": category,
        "response": final_response,
    }


def process_queries_batch(batch: Dict[str, Any]) -> Dict[str, List[Any]]:
    """Process a batch of queries through the conditional routing system."""
    # Get handles once for the entire batch
    handles = {
        "router": serve.get_deployment_handle(ROUTER_DEPLOYMENT_NAME, ROUTER_APP_NAME).options(stream=True),
        "weather": serve.get_deployment_handle(WEATHER_DEPLOYMENT_NAME, WEATHER_APP_NAME).options(stream=True),
        "science": serve.get_deployment_handle(SCIENCE_DEPLOYMENT_NAME, SCIENCE_APP_NAME).options(stream=True),
        "unknown": serve.get_deployment_handle(UNKNOWN_DEPLOYMENT_NAME, UNKNOWN_APP_NAME).options(stream=True),
    }

    ids = batch["id"]
    queries = batch["query"]

    results = {"id": [], "query": [], "category": [], "response": []}

    for row_id, query in zip(ids, queries):
        result = process_query(query, row_id, handles)
        results["id"].append(result["id"])
        results["query"].append(result["query"])
        results["category"].append(result["category"])
        results["response"].append(result["response"])

    return results


def create_synthetic_dataset() -> ray.data.Dataset:
    """Create a synthetic dataset of user queries."""
    queries = [
        "What's the weather like in Paris?",
        "Explain quantum physics simply.",
        "What is the capital of France?",
        "Will it rain tomorrow in San Francisco?",
        "How does photosynthesis work?",
        "What's the temperature in Tokyo right now?",
        "What is the theory of relativity?",
        "Tell me about the weather in New York.",
    ]
    
    data = [{"id": i, "query": query} for i, query in enumerate(queries)]
    return ray.data.from_items(data)


if __name__ == "__main__":
    print("=" * 80)
    print("Conditional Routing Multi-Agent System")
    print("=" * 80)
    print()
    
    # First, shut down ALL existing Ray Serve applications to free up GPUs
    # This is necessary because previous runs may have left deployments running
    print("Shutting down any existing Ray Serve applications...")
    try:
        serve.shutdown()
        print("✓ All existing applications shut down")
    except Exception as e:
        print(f"  Note: {e}")
    print()
    
    # Build and deploy Router Agent
    router_llm_app = build_llm_deployment(
        llm_config,
        name_prefix="",
        override_serve_options=dict(name=ROUTER_DEPLOYMENT_NAME)
    )
    router_app = serve.run(router_llm_app, name=ROUTER_APP_NAME, route_prefix="/router")
    print(f"✓ Router Agent deployed: {ROUTER_APP_NAME}")
    
    # Build and deploy Weather Agent
    weather_llm_app = build_llm_deployment(
        llm_config,
        name_prefix="",
        override_serve_options=dict(name=WEATHER_DEPLOYMENT_NAME)
    )
    weather_app = serve.run(weather_llm_app, name=WEATHER_APP_NAME, route_prefix="/weather")
    print(f"✓ Weather Agent deployed: {WEATHER_APP_NAME}")
    
    # Build and deploy Science Agent
    science_llm_app = build_llm_deployment(
        llm_config,
        name_prefix="",
        override_serve_options=dict(name=SCIENCE_DEPLOYMENT_NAME)
    )
    science_app = serve.run(science_llm_app, name=SCIENCE_APP_NAME, route_prefix="/science")
    print(f"✓ Science Agent deployed: {SCIENCE_APP_NAME}")
    
    # Build and deploy Unknown Agent
    unknown_llm_app = build_llm_deployment(
        llm_config,
        name_prefix="",
        override_serve_options=dict(name=UNKNOWN_DEPLOYMENT_NAME)
    )
    unknown_app = serve.run(unknown_llm_app, name=UNKNOWN_APP_NAME, route_prefix="/unknown")
    print(f"✓ Unknown Agent deployed: {UNKNOWN_APP_NAME}")
    
    print()
    print("=" * 80)
    print("Deployments:")
    print(f"  - Router Agent: {ROUTER_APP_NAME}")
    print(f"  - Weather Agent: {WEATHER_APP_NAME}")
    print(f"  - Science Agent: {SCIENCE_APP_NAME}")
    print(f"  - Unknown Agent: {UNKNOWN_APP_NAME}")
    print()
    
    # Create dataset
    print("Creating queries dataset...")
    dataset = create_synthetic_dataset()
    print(f"Dataset created with {dataset.count()} queries")
    print()
    
    # Process queries through conditional routing
    print("Processing queries through conditional routing system...")
    print("Each query will be:")
    print("  1. Classified by Router Agent")
    print("  2. Routed to appropriate specialized agent")
    print("  3. Responded to by specialized agent")
    print()
    
    # Process using Ray Data map_batches
    results = dataset.map_batches(process_queries_batch)
    
    # Collect results
    print()
    print("=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print()
    
    for result in results.iter_rows():
        print("=" * 80)
        print(f"Query {result['id']}: {result['query']}")
        print("=" * 80)
        print()
        print(f"📋 Category: {result['category']}")
        print(f"✅ Response ({len(result['response'])} chars):")
        print("-" * 80)
        print(result['response'])
        print()
    
    print()
    print("=" * 80)
    print("All queries processed!")
    print("=" * 80)

