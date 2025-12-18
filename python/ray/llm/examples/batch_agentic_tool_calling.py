"""
Agentic Batch Inference with Tool Calling

This example demonstrates how to perform batch inference with agentic tool calling using:
1. LLM Service: Deployed via build_llm_deployment for generating tool calls
2. Tool Service: Deployed as a Ray Serve deployment for executing tools
3. Batch Processing: Uses ServeDeploymentProcessorConfig to process a synthetic dataset

The workflow:
1. LLM generates tool calls for each query in the batch
2. Tool calls are extracted from LLM responses
3. Tools are executed via the Ray Serve tool deployment
4. Results are collected and displayed

For true multi-turn agentic behavior, you could extend this to:
- Make follow-up LLM calls with tool results
- Chain multiple tool calls
- Use an agent orchestrator (e.g., LangGraph) for complex workflows
"""

import ray
from ray import serve
from ray.data.llm import ServeDeploymentProcessorConfig, build_processor
from ray.serve.llm import LLMConfig, build_llm_deployment
from ray.serve.llm.openai_api_models import ChatCompletionRequest, CompletionRequest
import json
import copy
from typing import Dict, Any

# Deploy the tool as a Ray Serve deployment
@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_cpus": 0.5}
)
class WeatherToolDeployment:
    """Ray Serve deployment for weather tool execution."""
    
    def get_weather(self, location: str, unit: str) -> str:
        """Get the current weather in a given location.
        
        Args:
            location: City and state, e.g., 'San Francisco, CA'
            unit: Temperature unit, either 'celsius' or 'fahrenheit'
            
        Returns:
            Weather information string
        """
        # In a real implementation, this would call an actual weather API
        return f"The weather in {location} is 72 degrees {unit} with sunny skies."
    
    def __call__(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tool execution requests.
        
        Args:
            request: Dictionary with 'tool_name' and 'tool_args'
            
        Returns:
            Dictionary with 'result' containing the tool execution result
        """
        tool_name = request.get("tool_name")
        tool_args = request.get("tool_args", {})
        
        if tool_name == "get_weather":
            result = self.get_weather(
                location=tool_args.get("location"),
                unit=tool_args.get("unit")
            )
            return {"result": result}
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

# Define tools schema for the LLM
# Ensure tools are plain Python dicts/lists (not numpy arrays) for JSON serialization
tools_raw = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City and state, e.g., 'San Francisco, CA'"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location", "unit"]
        }
    }
}]

# Convert tools to ensure they're JSON-serializable (deep copy and JSON round-trip)
# This ensures no numpy arrays or other non-serializable types
# Create a fresh copy to avoid any potential numpy contamination
tools = json.loads(json.dumps(copy.deepcopy(tools_raw)))
# Verify tools are plain Python types
assert isinstance(tools, list), "Tools must be a list"
assert all(isinstance(t, dict) for t in tools), "Each tool must be a dict"

# Configure the LLM
llm_config = LLMConfig(
    model_loading_config=dict(
        # The name your clients will use in the OpenAI-compatible API.
        model_id="Qwen/Qwen3-4B-Instruct-2507-FP8",
        # Hugging Face repo to pull from.
        model_source="Qwen/Qwen3-4B-Instruct-2507-FP8",
    ),
    # L4 (Ada) is FP8-friendly. Prefer H100 for best FP8 throughput.
    accelerator_type="L4",
    deployment_config=dict(
        autoscaling_config=dict(
            num_replicas=1,  # use 1 replica for now
        )
    ),
    # vLLM engine flags.
    engine_kwargs=dict(
        # Qwen3 supports 262,144 context natively; but you need a GPU with large memory to serve.
        max_model_len=65536,
        # Qwen models use custom chat templates; needed for some Hugging Face repos.
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        enable_auto_tool_choice=True,
        tool_call_parser="hermes",
    ),
)

# Set up deployment names
APP_NAME = "agentic_tool_calling_app"
DEPLOYMENT_NAME = "Qwen--Qwen3-4B-Instruct-2507-FP8"
TOOL_APP_NAME = "weather_tool_app"
TOOL_DEPLOYMENT_NAME = "WeatherToolDeployment"
override_serve_options = dict(name=DEPLOYMENT_NAME)

# Deploy the tool service with a specific route prefix
tool_app = serve.run(
    WeatherToolDeployment.bind(),
    name=TOOL_APP_NAME,
    route_prefix="/tools"  # Use /tools route prefix to avoid conflict
)
tool_handle = serve.get_deployment_handle(TOOL_DEPLOYMENT_NAME, TOOL_APP_NAME)

# Build and deploy the LLM deployment
# Use name_prefix="" to avoid the default "LLMServer:" prefix
llm_app = build_llm_deployment(
    llm_config, 
    name_prefix="",  # Disable default prefix
    override_serve_options=override_serve_options
)
app = serve.run(llm_app, name=APP_NAME, route_prefix="/llm")  # Use /llm route prefix

# Configure the batch inference processor
config = ServeDeploymentProcessorConfig(
    deployment_name=DEPLOYMENT_NAME,
    app_name=APP_NAME,
    dtype_mapping={
        "ChatCompletionRequest": ChatCompletionRequest,
    },
    concurrency=1,
    batch_size=16,
)

# Create synthetic dataset with different weather queries
def create_synthetic_dataset():
    """Create a synthetic dataset with weather queries."""
    locations = [
        "San Francisco, CA",
        "New York, NY",
        "Los Angeles, CA",
        "Chicago, IL",
        "Miami, FL",
        "Seattle, WA",
        "Boston, MA",
        "Denver, CO",
    ]
    units = ["celsius", "fahrenheit"]
    
    # Create dataset with varying queries
    data = []
    for i, location in enumerate(locations):
        unit = units[i % len(units)]
        data.append({
            "id": i,
            "location": location,
            "unit": unit,
            "query": f"What's the weather like in {location}?",
        })
    
    return ray.data.from_items(data)

# Build the processor for agentic tool calling with Chat Completions API
# Create a function to sanitize tools for each request to avoid ndarray serialization issues
def sanitize_tools():
    """Create a fresh, sanitized copy of tools for each request."""
    # Create tools fresh each time to avoid any numpy contamination
    # Explicitly use list() constructor to ensure enum is a plain Python list
    tools_fresh = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City and state, e.g., 'San Francisco, CA'"},
                    "unit": {"type": "string", "enum": list(["celsius", "fahrenheit"])}  # Explicit list() to avoid numpy
                },
                "required": list(["location", "unit"])  # Explicit list() to avoid numpy
            }
        }
    }]
    # JSON round-trip to ensure pure Python types (no numpy arrays)
    sanitized = json.loads(json.dumps(tools_fresh))
    # Double-check: recursively convert any remaining numpy arrays
    def ensure_python_types(obj):
        if isinstance(obj, dict):
            return {k: ensure_python_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [ensure_python_types(item) for item in obj]
        return obj
    return ensure_python_types(sanitized)

processor = build_processor(
    config,
    preprocess=lambda row: dict(
        # Preserve original row data
        id=row["id"],
        query=row["query"],
        location=row["location"],
        unit=row["unit"],
        # Request configuration for serve deployment - using Chat Completions API for tool calling
        method="chat",
        dtype="ChatCompletionRequest",
        request_kwargs=dict(
            model=llm_config.model_id,
            messages=[
                {"role": "user", "content": row["query"]}
            ],
            # Create fresh tools for each request to avoid serialization issues
            tools=sanitize_tools(),
            tool_choice="auto",
            stream=False,
        ),
    ),
    postprocess=lambda row: dict(
        # Original row data
        id=row.get("id", ""),
        query=row.get("query", ""),
        location=row.get("location", ""),
        unit=row.get("unit", ""),
        # Extract tool call information from response
        choices=row.get("choices", []),
        tool_calls=row.get("choices", [{}])[0].get("message", {}).get("tool_calls", []) if row.get("choices") else [],
        response_content=row.get("choices", [{}])[0].get("message", {}).get("content", "") if row.get("choices") else "",
        error=row.get("error"),
        raw_response=row,
    ),
    preprocess_map_kwargs={"batch_format": "python"},
)

# Create and process the synthetic dataset
print("Creating synthetic dataset...")
ds = create_synthetic_dataset()
print(f"Dataset created with {ds.count()} rows")

print("\nProcessing dataset through agentic batch inference...")
ds = processor(ds)
ds = ds.materialize()

# Process results and execute tool calls via Ray Serve
print("\nResults:")
results = ds.take_all()
print(results)

# for result in results:
#     print(f"\nQuery: {result.get('query', 'N/A')}")
#     print(f"Location: {result.get('location', 'N/A')}, Unit: {result.get('unit', 'N/A')}")
    
#     # Check for errors first
#     if result.get("error"):
#         error_msg = result.get("error")
#         if isinstance(error_msg, dict):
#             error_msg = error_msg.get("message", str(error_msg))
#         print(f"Error in request: {error_msg}")
#         continue
    
#     # Debug: print raw response structure if needed
#     if not result.get("choices"):
#         print(f"Warning: No choices in response. Raw keys: {list(result.keys())}")
#         if "raw_response" in result:
#             print(f"Raw response keys: {list(result.get('raw_response', {}).keys())}")
#         continue
    
#     # Extract message from first choice
#     message = result.get("choices", [{}])[0].get("message", {})
#     tool_calls = message.get("tool_calls", [])
#     response_content = message.get("content", "")
    
#     # Check if tool was called
#     if tool_calls:
#         tool_call = tool_calls[0]
#         function_info = tool_call.get("function", {})
#         function_name = function_info.get("name", "")
#         arguments_str = function_info.get("arguments", "{}")
        
#         try:
#             arguments = json.loads(arguments_str) if isinstance(arguments_str, str) else arguments_str
#         except (json.JSONDecodeError, TypeError):
#             arguments = {}
#             print(f"Warning: Could not parse tool arguments: {arguments_str}")
        
#         print(f"Tool called: {function_name}")
#         print(f"Arguments: {arguments}")
        
#         # Execute the tool via Ray Serve deployment
#         if function_name == "get_weather":
#             tool_request = {
#                 "tool_name": "get_weather",
#                 "tool_args": arguments
#             }
#             try:
#                 tool_response = ray.get(tool_handle.remote(tool_request))
#                 tool_result = tool_response.get("result", "Tool execution failed")
#                 print(f"Tool result (from Ray Serve): {tool_result}")
#             except Exception as e:
#                 print(f"Error executing tool: {e}")
#     else:
#         print(f"Response: {response_content if response_content else 'No response content'}")