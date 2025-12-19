"""
Agentic Multi-Turn Rollout with Tool Calling

This example demonstrates multi-turn agentic tool calling where the agent repeatedly
invokes tools until its goal is completed. The agent can make multiple tool calls
in sequence, with each tool result feeding back into the conversation for the next turn.

The workflow:
1. LLM Service: Deployed via build_llm_deployment for generating tool calls
2. Tool Service: Deployed as a Ray Serve deployment for executing tools
3. Multi-Turn Loop: Agent repeatedly calls tools until goal completion or max turns

Multi-turn process:
1. Initial LLM call with user query and available tools
2. Extract tool calls from LLM response
3. Execute tools via Ray Serve tool deployment
4. Add tool results to conversation history
5. Make follow-up LLM call with updated conversation history
6. Repeat until no more tool calls or max turns reached

This pattern is useful for:
- Booking appointments using a calendar API
- Retrieving real-time stock prices via a financial API
- Searching a vector database for relevant documents (RAG)
- Controlling smart home devices
- Executing code snippets
"""

import ray
# Initialize Ray with runtime_env that excludes large files to avoid package size issues
try:
    ray.init(
        ignore_reinit_error=True,
        runtime_env={
            "excludes": [
                "*.pyc", "*.pyo", "__pycache__", "*.so", "*.dylib", 
                "*.egg-info", ".git", "*.md", "*.txt", "tests/", 
                "examples/", "docs/", "*.log", "*.ipynb"
            ]
        }
    )
except Exception:
    # If already initialized, continue
    pass

from ray import serve
from ray.data.llm import ServeDeploymentProcessorConfig, build_processor
from ray.serve.llm import LLMConfig, build_llm_deployment
from ray.serve.llm.openai_api_models import ChatCompletionRequest, ChatCompletionResponse
import json
from typing import Dict, Any, List, Optional

# Note: The ValidatorIterator serialization issue is now fixed in
# ray/python/ray/llm/_internal/batch/stages/serve_deployment_stage.py
# The ServeDeploymentStageUDF.generate_async method now automatically
# sanitizes ChatCompletionRequest objects before sending them.

# Deploy the tool as a Ray Serve deployment
@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_cpus": 0.5}
)
class CalendarToolDeployment:
    """Ray Serve deployment for calendar booking tool execution.
    
    This tool simulates a calendar API that requires multiple steps:
    1. Check availability for a time slot
    2. Book an appointment if available
    """
    
    def __init__(self):
        # Simulate a calendar with some existing bookings
        self.bookings = {
            "2024-01-15": ["10:00", "14:00"],
            "2024-01-16": ["09:00", "15:00"],
        }
    
    def check_availability(self, date: str, time: str) -> str:
        """Check if a time slot is available.
        
        Args:
            date: Date in YYYY-MM-DD format
            time: Time in HH:MM format (24-hour)
            
        Returns:
            Availability status string
        """
        if date in self.bookings and time in self.bookings[date]:
            return f"Time slot {date} at {time} is NOT available (already booked)."
        return f"Time slot {date} at {time} is available."
    
    def book_appointment(self, date: str, time: str, title: str) -> str:
        """Book an appointment at a specific time.
        
        Args:
            date: Date in YYYY-MM-DD format
            time: Time in HH:MM format (24-hour)
            title: Title/description of the appointment
            
        Returns:
            Booking confirmation string
        """
        if date in self.bookings and time in self.bookings[date]:
            return f"ERROR: Cannot book {date} at {time} - already booked."
        
        if date not in self.bookings:
            self.bookings[date] = []
        self.bookings[date].append(time)
        return f"Successfully booked appointment '{title}' on {date} at {time}."
    
    def list_available_slots(self, date: str) -> str:
        """List available time slots for a given date.
        
        Args:
            date: Date in YYYY-MM-DD format
            
        Returns:
            String listing available slots
        """
        booked = self.bookings.get(date, [])
        all_slots = [f"{h:02d}:00" for h in range(9, 18)]  # 9 AM to 5 PM
        available = [slot for slot in all_slots if slot not in booked]
        if available:
            return f"Available slots on {date}: {', '.join(available)}"
        return f"No available slots on {date} (all booked)."
    
    def get_calendar_state(self) -> Dict[str, Any]:
        """Get the current calendar state showing all bookings and availability.
        
        Returns:
            Dictionary with dates, their bookings, and available slots
        """
        all_slots = [f"{h:02d}:00" for h in range(9, 18)]  # 9 AM to 5 PM
        result = {}
        # Include dates we're interested in
        dates_of_interest = ["2024-01-15", "2024-01-16"]
        for date in dates_of_interest:
            booked = sorted(self.bookings.get(date, []))
            available = [slot for slot in all_slots if slot not in booked]
            result[date] = {
                "booked": booked,
                "available": available
            }
        return result
    
    def __call__(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tool execution requests.
        
        Args:
            request: Dictionary with 'tool_name' and 'tool_args'
            
        Returns:
            Dictionary with 'result' containing the tool execution result
        """
        tool_name = request.get("tool_name")
        tool_args = request.get("tool_args", {})
        
        if tool_name == "check_availability":
            result = self.check_availability(
                date=tool_args.get("date"),
                time=tool_args.get("time")
            )
            return {"result": result}
        elif tool_name == "book_appointment":
            result = self.book_appointment(
                date=tool_args.get("date"),
                time=tool_args.get("time"),
                title=tool_args.get("title", "Meeting")
            )
            return {"result": result}
        elif tool_name == "list_available_slots":
            result = self.list_available_slots(
                date=tool_args.get("date")
            )
            return {"result": result}
        elif tool_name == "get_calendar_state":
            return {"result": self.get_calendar_state()}
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

# Define tools schema for the LLM
# Ensure tools are plain Python dicts/lists (not numpy arrays) for JSON serialization
tools_raw = [
    {
        "type": "function",
        "function": {
            "name": "check_availability",
            "description": "Check if a specific time slot is available for booking",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date in YYYY-MM-DD format"},
                    "time": {"type": "string", "description": "Time in HH:MM format (24-hour)"}
                },
                "required": ["date", "time"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "book_appointment",
            "description": "Book an appointment at a specific date and time",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date in YYYY-MM-DD format"},
                    "time": {"type": "string", "description": "Time in HH:MM format (24-hour)"},
                    "title": {"type": "string", "description": "Title or description of the appointment"}
                },
                "required": ["date", "time", "title"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_available_slots",
            "description": "List all available time slots for a given date",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date in YYYY-MM-DD format"}
                },
                "required": ["date"]
            }
        }
    }
]

# Convert tools to ensure they're JSON-serializable (JSON round-trip)
# This ensures no numpy arrays or other non-serializable types
tools = json.loads(json.dumps(tools_raw))
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
TOOL_APP_NAME = "calendar_tool_app"
TOOL_DEPLOYMENT_NAME = "CalendarToolDeployment"
override_serve_options = dict(name=DEPLOYMENT_NAME)

# Deploy the tool service with a specific route prefix
# Use a unique route prefix to avoid conflicts with existing deployments
tool_app = serve.run(
    CalendarToolDeployment.bind(),
    name=TOOL_APP_NAME,
    route_prefix="/calendar_tools"  # Use unique route prefix
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


def execute_tool_call_sync(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a single tool call via Ray Serve (synchronous for batch processing).
    
    Args:
        tool_call: Tool call dictionary with 'id', 'function' containing 'name' and 'arguments'
        
    Returns:
        Dictionary with tool execution result
    """
    function = tool_call.get("function", {})
    tool_name = function.get("name")
    tool_args_str = function.get("arguments", "{}")
    
    # Parse JSON arguments
    try:
        tool_args = json.loads(tool_args_str)
    except json.JSONDecodeError:
        tool_args = {}
    
    # Execute tool via Ray Serve (synchronous)
    request = {
        "tool_name": tool_name,
        "tool_args": tool_args
    }
    result = tool_handle.remote(request).result()
    return result


def get_calendar_state() -> Dict[str, Any]:
    """Get the current calendar state from the tool deployment."""
    request = {"tool_name": "get_calendar_state", "tool_args": {}}
    result = tool_handle.remote(request).result()
    return result.get("result", {})


def display_calendar_state(title: str = "Calendar State"):
    """Display the current calendar state in a formatted way."""
    state = get_calendar_state()
    print(f"\n{'=' * 60}")
    print(f"📅 {title}")
    print("=" * 60)
    for date, info in sorted(state.items()):
        booked = info.get("booked", [])
        available = info.get("available", [])
        print(f"\n  {date}:")
        print(f"    ❌ Booked slots:    {', '.join(booked) if booked else '(none)'}")
        print(f"    ✅ Available slots: {', '.join(available) if available else '(none)'}")
    print("=" * 60)


# Create synthetic dataset with different booking queries
def create_synthetic_dataset():
    """Create a synthetic dataset with booking queries."""
    queries = [
        "Book me an appointment for a team meeting on 2024-01-15 at 10:00. If it's not available, book the next available slot.",
        "I need to schedule a doctor's appointment. Check what times are available on 2024-01-16",
        "Can you find an available slot on 2024-01-15 and book it for 'Project Review'?",
    ]
    
    data = []
    for i, query in enumerate(queries):
        data.append({
            "id": i,
            "query": query,
        })
    
    return ray.data.from_items(data)


# Get handle to LLM deployment for direct calls (with stream=True for generator methods)
llm_handle = serve.get_deployment_handle(DEPLOYMENT_NAME, APP_NAME).options(stream=True)


def sanitize_messages_for_request(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sanitize messages to ensure tool_calls are plain lists, avoiding ValidatorIterator.
    
    Args:
        messages: List of message dicts
        
    Returns:
        Sanitized list of message dicts
    """
    sanitized = []
    for msg in messages:
        if not isinstance(msg, dict):
            sanitized.append(msg)
            continue
            
        msg_copy = msg.copy()
        # Ensure tool_calls is a plain list of dicts
        if "tool_calls" in msg_copy and msg_copy["tool_calls"] is not None:
            tool_calls = msg_copy["tool_calls"]
            # Force to list if it's an iterator
            if not isinstance(tool_calls, list):
                try:
                    tool_calls = list(tool_calls)
                except (TypeError, ValueError):
                    tool_calls = []
            # Ensure each tool_call is a plain dict via JSON round-trip
            sanitized_tool_calls = []
            for tc in tool_calls:
                if hasattr(tc, "model_dump"):
                    tc = tc.model_dump(mode="python")
                if isinstance(tc, dict):
                    tc = json.loads(json.dumps(tc))
                sanitized_tool_calls.append(tc)
            msg_copy["tool_calls"] = sanitized_tool_calls
        sanitized.append(msg_copy)
    
    # Final JSON round-trip to ensure everything is plain
    return json.loads(json.dumps(sanitized))


def call_llm_sync(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Call the LLM deployment synchronously with the given messages.
    
    Args:
        messages: List of message dicts for the conversation
        
    Returns:
        The LLM response as a dict
    """
    # Import the sanitization function from serve_deployment_stage
    from ray.llm._internal.batch.stages.serve_deployment_stage import (
        _sanitize_request_tool_calls,
    )
    
    # Sanitize messages to prevent ValidatorIterator serialization issues
    sanitized_messages = sanitize_messages_for_request(messages)
    
    # Create the request using normal constructor (tools need to be Pydantic models)
    # The messages are already sanitized to be plain dicts
    request = ChatCompletionRequest(
        model=llm_config.model_id,
        messages=sanitized_messages,
        tools=tools,  # Tools are plain dicts, Pydantic will convert them
        tool_choice="auto",
        stream=False,
    )
    
    # Sanitize the request using the same function as serve_deployment_stage
    # This properly handles ValidatorIterator issues
    try:
        request = _sanitize_request_tool_calls(request, ChatCompletionRequest)
    except (ImportError, TypeError, AttributeError) as e:
        # If sanitization fails, try the ingress sanitization as fallback
        try:
            from ray.llm._internal.serve.core.ingress.ingress import (
                _sanitize_chat_completion_request,
            )
            request = _sanitize_chat_completion_request(request)
        except (ImportError, TypeError, AttributeError):
            pass
    
    # Call the LLM deployment - returns an async generator
    # Use the stream handle and get the first (only) result for non-streaming
    response_gen = llm_handle.chat.remote(request)
    
    # Iterate through the generator to get the response
    # For stream=False requests, the generator yields a single complete response
    response = None
    for chunk in response_gen:
        response = chunk
    
    if response is None:
        raise RuntimeError("No response received from LLM deployment")
    
    # Convert to dict if it's a Pydantic model
    if hasattr(response, 'model_dump'):
        return response.model_dump()
    return response


def process_row_multi_turn(row: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single row through multiple turns until completion.
    
    Each row independently executes its own multi-turn loop:
    1. Call LLM with current messages
    2. If response has tool calls, execute them and add to messages
    3. Repeat until no more tool calls or max turns reached
    
    Args:
        row: Input row with 'id' and 'query'
        
    Returns:
        Completed row with full conversation history and final response
    """
    max_turns = 10
    
    # Initialize state for this row
    row_id = row.get("id", "")
    query = row.get("query", "")
    messages = [{"role": "user", "content": query}]
    conversation_history = []
    total_tool_calls = 0
    final_response = ""
    
    for turn_num in range(1, max_turns + 1):
        # Call LLM
        try:
            response = call_llm_sync(messages)
        except Exception as e:
            # Handle LLM call errors
            return {
                "id": row_id,
                "original_query": query,
                "turn": turn_num,
                "messages": messages,
                "conversation_history": conversation_history,
                "total_tool_calls": total_tool_calls,
                "response_content": f"Error: {str(e)}",
                "error": str(e),
            }
        
        # Extract response content and tool calls
        choices = response.get("choices", [])
        if not choices:
            break
            
        message = choices[0].get("message", {})
        content = message.get("content", "") or ""
        tool_calls = message.get("tool_calls", []) or []
        
        # Sanitize tool_calls to plain dicts
        tool_calls = json.loads(json.dumps(tool_calls)) if tool_calls else []
        
        # Add assistant message to conversation
        assistant_message = {
            "role": "assistant",
            "content": content if content else None,
        }
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
        messages.append(assistant_message)
        
        # Record in conversation history
        conversation_history.append({
            "turn": turn_num,
            "type": "assistant",
            "content": content,
            "tool_calls": tool_calls
        })
        
        # If no tool calls, we're done
        if not tool_calls:
            final_response = content
            break
        
        # Execute all tool calls
        tool_results = []
        for tool_call in tool_calls:
            result = execute_tool_call_sync(tool_call)
            tool_result_content = result.get("result", str(result))
            
            # Add tool result to conversation
            tool_message = {
                "role": "tool",
                "content": tool_result_content,
                "tool_call_id": tool_call.get("id")
            }
            messages.append(tool_message)
            
            tool_results.append({
                "tool_call_id": tool_call.get("id"),
                "tool_name": tool_call.get("function", {}).get("name"),
                "result": tool_result_content
            })
            total_tool_calls += 1
        
        # Record tool results in history
        conversation_history.append({
            "turn": turn_num,
            "type": "tool_results",
            "results": tool_results
        })
        
        # Update final response (in case we hit max turns)
        final_response = content
    
    # Return completed row
    return {
        "id": row_id,
        "original_query": query,
        "turn": turn_num,
        "messages": json.loads(json.dumps(messages)),  # Sanitize
        "conversation_history": conversation_history,
        "total_tool_calls": total_tool_calls,
        "response_content": final_response,
        "error": None,
    }

# Create and process the synthetic dataset
print("Creating synthetic dataset...")
ds = create_synthetic_dataset()
print(f"Dataset created with {ds.count()} rows")

# Display initial calendar state
display_calendar_state("INITIAL CALENDAR STATE (Before Processing)")

print("\nProcessing dataset through multi-turn agentic batch inference...")
print("Each row will independently execute its own multi-turn loop until completion.")
print("Materialization happens only at the end.\n")

# Process all rows - each row handles its own multi-turn loop independently
# This is a single map operation where each row executes turns until:
# - No more tool calls are returned, or
# - Max turns (10) is reached
ds = ds.map(process_row_multi_turn)

# Materialize only once at the end
print("Executing multi-turn conversations for all rows...")
ds = ds.materialize()
print("All conversations completed.")

# Final results
print("\n" + "=" * 80)
print("Final Results:")
print("=" * 80)
results = ds.take_all()
for result in results:
    print(f"\nQuery: {result.get('original_query', result.get('query', ''))}")
    print(f"Total Turns: {result.get('turn', 1) - 1}")
    print(f"Total Tool Calls: {result.get('total_tool_calls', 0)}")
    final_response = result.get("response_content", "")
    if final_response:
        print(f"Final Response: {final_response[:200]}...")
    else:
        print("Final Response: (No final response - may have hit max turns)")
    
    # Show full conversation history
    history = result.get("conversation_history", [])
    if history:
        print(f"\nConversation History ({len(history)} entries):")
        print("-" * 60)
        for entry in history:
            turn = entry.get('turn')
            entry_type = entry.get('type')
            if entry_type == 'assistant':
                content = entry.get('content', '')
                tool_calls = entry.get('tool_calls', [])
                if tool_calls:
                    tool_names = [tc.get('function', {}).get('name', 'unknown') for tc in tool_calls]
                    print(f"  [Turn {turn}] ASSISTANT calls tools: {', '.join(tool_names)}")
                    for tc in tool_calls:
                        func = tc.get('function', {})
                        print(f"           -> {func.get('name')}({func.get('arguments', '{}')})")
                elif content:
                    # Truncate long responses for readability
                    display_content = content[:300] + "..." if len(content) > 300 else content
                    print(f"  [Turn {turn}] ASSISTANT: {display_content}")
            elif entry_type == 'tool_results':
                results = entry.get('results', [])
                print(f"  [Turn {turn}] TOOL RESULTS:")
                for tr in results:
                    tool_name = tr.get('tool_name', 'unknown')
                    result_text = tr.get('result', '')
                    # Truncate long results for readability
                    display_result = result_text[:200] + "..." if len(result_text) > 200 else result_text
                    print(f"           <- {tool_name}: {display_result}")
        print("-" * 60)

# Display final calendar state after all processing
display_calendar_state("FINAL CALENDAR STATE (After Processing)")

# Summary of changes
print("\n" + "=" * 60)
print("📊 SUMMARY: Booking Changes")
print("=" * 60)
print("Initial state had:")
print("  2024-01-15: 10:00 and 14:00 were pre-booked")
print("  2024-01-16: 09:00 and 15:00 were pre-booked")
print("\nAfter agentic processing, check the final state above to see")
print("which new appointments were booked by the LLM agent!")
print("=" * 60)