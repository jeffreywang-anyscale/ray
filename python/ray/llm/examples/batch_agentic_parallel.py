"""
Parallel Multi-Agent Coding Assistant

This example demonstrates parallel agent coordination where:
1. Plan Agent: Breaks down coding tasks into focused subtasks and synthesizes fixes
2. Explore Agent 1: Analyzes code implementation, structure, and logic
3. Explore Agent 2: Analyzes code rendering, UI behavior, and presentation

The workflow:
1. Plan Agent receives a coding task and breaks it into 2 subtasks
2. Two Explore Agents work in parallel on different aspects
3. Plan Agent synthesizes exploration results into a suggested fix

This pattern is useful for:
- Code review and bug fixing
- Task decomposition and parallel analysis
- Multi-perspective code understanding
- Collaborative agent systems
"""

import ray
import os

# Initialize Ray WITHOUT runtime_env to avoid packaging the working directory
# The Ray cluster already has all necessary code installed
# Setting working_dir to None avoids the package size limit issue
try:
    # Check if already connected
    if ray.is_initialized():
        print("Ray already initialized, reusing connection")
    else:
        # Initialize without working_dir to avoid packaging
        # The cluster should already have the necessary code
        ray.init(
            ignore_reinit_error=True,
            # Do NOT set runtime_env here - let it inherit from the cluster
        )
        print("Ray initialized successfully")
except Exception as e:
    print(f"Ray init note: {e}")
    # Continue anyway - might already be connected
    pass

from ray import serve
from ray.serve.llm import LLMConfig, build_llm_deployment
from ray.serve.llm.openai_api_models import ChatCompletionRequest
import json
from typing import Dict, Any, List, Optional

# Configure the LLM (same model for all agents, different system prompts)
llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="Qwen/Qwen3-4B-Instruct-2507-FP8",
        model_source="Qwen/Qwen3-4B-Instruct-2507-FP8",
    ),
    accelerator_type="L4",
    deployment_config=dict(
        autoscaling_config=dict(
            num_replicas=1,
        )
    ),
    engine_kwargs=dict(
        max_model_len=65536,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
    ),
)

# Set up deployment names
PLAN_APP_NAME = "plan_agent_app"
PLAN_DEPLOYMENT_NAME = "PlanAgent"
EXPLORE1_APP_NAME = "explore_agent_1_app"
EXPLORE1_DEPLOYMENT_NAME = "ExploreAgent1"
EXPLORE2_APP_NAME = "explore_agent_2_app"
EXPLORE2_DEPLOYMENT_NAME = "ExploreAgent2"

# System prompts for each agent
PLAN_AGENT_SYSTEM_PROMPT = """You are a coding task planner and synthesizer. Your role is to:
1. Break down coding tasks into 2 focused subtasks that can be explored in parallel
2. Synthesize exploration results from multiple agents into actionable code fixes

When breaking down tasks:
- Create exactly 2 subtasks
- First subtask should focus on implementation, structure, logic, and code flow
- Second subtask should focus on rendering, UI behavior, styling, and presentation
- Each subtask should be specific and actionable

When synthesizing results:
- Combine findings from both exploration agents
- Identify the root cause of the issue
- Provide a clear, actionable fix with code changes
- Explain why the fix addresses the problem

Output your responses in JSON format when possible."""

EXPLORE_AGENT_1_SYSTEM_PROMPT = """You are a code implementation explorer. Your role is to:
1. Analyze code structure, logic, data flow, and implementation patterns
2. Identify issues related to component structure, state management, prop handling, and logic
3. Focus on HOW the code works, not how it looks

When analyzing code:
- Examine component structure and organization
- Check prop handling and data flow
- Identify logic errors or missing implementations
- Look for state management issues
- Check for proper error handling

Provide your findings in a structured format with:
- Analysis summary
- Issues found
- Recommendations"""

EXPLORE_AGENT_2_SYSTEM_PROMPT = """You are a code rendering explorer. Your role is to:
1. Analyze UI rendering, styling, visual behavior, and presentation logic
2. Identify issues related to display, styling, CSS, visual feedback, and user experience
3. Focus on HOW the code is presented, not the underlying logic

When analyzing code:
- Examine rendering logic and display behavior
- Check CSS classes, styling, and visual indicators
- Identify missing visual feedback or styling
- Look for rendering performance issues
- Check for accessibility and visual consistency

Provide your findings in a structured format with:
- Analysis summary
- Issues found
- Recommendations"""

# Build and deploy Plan Agent
# Note: Runtime environment packaging may fail if working directory is too large
# This is expected when already connected to a Ray cluster
try:
    plan_llm_app = build_llm_deployment(
        llm_config,
        name_prefix="",
        override_serve_options=dict(name=PLAN_DEPLOYMENT_NAME)
    )
    plan_app = serve.run(plan_llm_app, name=PLAN_APP_NAME, route_prefix="/plan")
    print(f"✓ Plan Agent deployed: {PLAN_APP_NAME}")
except Exception as e:
    print(f"⚠ Warning deploying Plan Agent: {e}")
    print("Attempting to use existing deployment...")
    # Try to get existing deployment
    try:
        plan_app = serve.get_app_handle(PLAN_APP_NAME)
        print(f"✓ Using existing Plan Agent: {PLAN_APP_NAME}")
    except Exception:
        raise RuntimeError(f"Failed to deploy or find Plan Agent: {e}")

# Build and deploy Explore Agent 1
try:
    explore1_llm_app = build_llm_deployment(
        llm_config,
        name_prefix="",
        override_serve_options=dict(name=EXPLORE1_DEPLOYMENT_NAME)
    )
    explore1_app = serve.run(explore1_llm_app, name=EXPLORE1_APP_NAME, route_prefix="/explore1")
    print(f"✓ Explore Agent 1 deployed: {EXPLORE1_APP_NAME}")
except Exception as e:
    print(f"⚠ Warning deploying Explore Agent 1: {e}")
    try:
        explore1_app = serve.get_app_handle(EXPLORE1_APP_NAME)
        print(f"✓ Using existing Explore Agent 1: {EXPLORE1_APP_NAME}")
    except Exception:
        raise RuntimeError(f"Failed to deploy or find Explore Agent 1: {e}")

# Build and deploy Explore Agent 2
try:
    explore2_llm_app = build_llm_deployment(
        llm_config,
        name_prefix="",
        override_serve_options=dict(name=EXPLORE2_DEPLOYMENT_NAME)
    )
    explore2_app = serve.run(explore2_llm_app, name=EXPLORE2_APP_NAME, route_prefix="/explore2")
    print(f"✓ Explore Agent 2 deployed: {EXPLORE2_APP_NAME}")
except Exception as e:
    print(f"⚠ Warning deploying Explore Agent 2: {e}")
    try:
        explore2_app = serve.get_app_handle(EXPLORE2_APP_NAME)
        print(f"✓ Using existing Explore Agent 2: {EXPLORE2_APP_NAME}")
    except Exception:
        raise RuntimeError(f"Failed to deploy or find Explore Agent 2: {e}")

# Get handles to all deployments
plan_handle = serve.get_deployment_handle(PLAN_DEPLOYMENT_NAME, PLAN_APP_NAME).options(stream=True)
explore1_handle = serve.get_deployment_handle(EXPLORE1_DEPLOYMENT_NAME, EXPLORE1_APP_NAME).options(stream=True)
explore2_handle = serve.get_deployment_handle(EXPLORE2_DEPLOYMENT_NAME, EXPLORE2_APP_NAME).options(stream=True)


def sanitize_messages_for_request(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sanitize messages to ensure tool_calls are plain lists, avoiding ValidatorIterator."""
    sanitized = []
    for msg in messages:
        if not isinstance(msg, dict):
            sanitized.append(msg)
            continue
            
        msg_copy = msg.copy()
        if "tool_calls" in msg_copy and msg_copy["tool_calls"] is not None:
            tool_calls = msg_copy["tool_calls"]
            if not isinstance(tool_calls, list):
                try:
                    tool_calls = list(tool_calls)
                except (TypeError, ValueError):
                    tool_calls = []
            sanitized_tool_calls = []
            for tc in tool_calls:
                if hasattr(tc, "model_dump"):
                    tc = tc.model_dump(mode="python")
                if isinstance(tc, dict):
                    tc = json.loads(json.dumps(tc))
                sanitized_tool_calls.append(tc)
            msg_copy["tool_calls"] = sanitized_tool_calls
        sanitized.append(msg_copy)
    
    return json.loads(json.dumps(sanitized))


def call_llm_sync(handle, messages: List[Dict[str, Any]], system_prompt: str = None) -> Dict[str, Any]:
    """Call an LLM deployment synchronously with the given messages."""
    # Import sanitization function
    _sanitize_request_tool_calls = None
    try:
        from ray.llm._internal.batch.stages.serve_deployment_stage import (
            _sanitize_request_tool_calls,
        )
    except (ImportError, AttributeError):
        # Fallback to ingress sanitization
        try:
            from ray.llm._internal.serve.core.ingress.ingress import (
                _sanitize_chat_completion_request,
            )
            _sanitize_request_tool_calls = lambda req, dtype: _sanitize_chat_completion_request(req) if isinstance(req, ChatCompletionRequest) else req
        except (ImportError, AttributeError):
            pass
    
    # Prepare messages with system prompt
    full_messages = []
    if system_prompt:
        full_messages.append({"role": "system", "content": system_prompt})
    full_messages.extend(messages)
    
    # Sanitize messages
    sanitized_messages = sanitize_messages_for_request(full_messages)
    
    # Create request
    request = ChatCompletionRequest(
        model=llm_config.model_id,
        messages=sanitized_messages,
        stream=False,
    )
    
    # Sanitize request
    if _sanitize_request_tool_calls:
        try:
            request = _sanitize_request_tool_calls(request, ChatCompletionRequest)
        except Exception:
            pass
    
    # Call LLM - handle returns a generator when stream=True
    response_gen = handle.chat.remote(request)
    
    # Get response - iterate through generator
    response = None
    try:
        for chunk in response_gen:
            response = chunk
    except Exception as e:
        # If generator iteration fails, try getting result directly
        if hasattr(response_gen, 'result'):
            response = response_gen.result()
        else:
            raise RuntimeError(f"Failed to get response from LLM: {e}")
    
    if response is None:
        raise RuntimeError("No response received from LLM deployment")
    
    # Convert to dict
    if hasattr(response, 'model_dump'):
        return response.model_dump()
    elif hasattr(response, 'dict'):
        return response.dict()
    return response


def parse_subtasks(plan_response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse subtasks from plan agent response. Raises exception if parsing fails."""
    choices = plan_response.get("choices", [])
    if not choices:
        raise ValueError("Plan response has no choices")
    
    content = choices[0].get("message", {}).get("content", "")
    if not content:
        raise ValueError("Plan response has no content")
    
    # Try to parse JSON from response
    import re
    json_match = re.search(r'\{.*\}', content, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            if "subtasks" in parsed:
                subtasks = parsed["subtasks"]
                if len(subtasks) != 2:
                    raise ValueError(f"Expected exactly 2 subtasks, got {len(subtasks)}")
                return subtasks
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from plan response: {e}")
    
    # Fallback: try to extract subtasks from text
    # Look for numbered or bulleted lists
    lines = content.split('\n')
    subtasks = []
    for line in lines:
        line = line.strip()
        if line and (line.startswith('-') or line.startswith('*') or 
                    line[0].isdigit() or 'subtask' in line.lower()):
            # Extract description
            desc = line.lstrip('-*0123456789. ').strip()
            if desc and len(desc) > 10:
                subtasks.append({
                    "description": desc,
                    "focus": "implementation" if len(subtasks) == 0 else "rendering"
                })
                if len(subtasks) >= 2:
                    break
    
    if len(subtasks) != 2:
        raise ValueError(f"Failed to extract exactly 2 subtasks from plan response. Found {len(subtasks)} subtasks. Content: {content[:200]}")
    
    return subtasks


def process_coding_task(row: Dict[str, Any]) -> Dict[str, Any]:
    """Process a coding task through the parallel agent system.
    
    Workflow:
    1. Plan Agent breaks down task into 2 subtasks
    2. Two Explore Agents work in parallel
    3. Plan Agent synthesizes results into a fix
    """
    task = row.get("task", "")
    code_context = row.get("code_context", "")
    row_id = row.get("id", "")
    
    print(f"\n{'='*80}")
    print(f"Processing Task {row_id}: {task[:60]}...")
    print(f"{'='*80}")
    
    # Step 1: Plan Agent breaks down the task
    print("\n[Step 1] Plan Agent: Breaking down task into subtasks...")
    plan_breakdown_prompt = f"""Break down this coding task into exactly 2 focused subtasks:

Task: {task}

Code Context:
{code_context}

Create 2 subtasks:
1. First subtask should focus on implementation, structure, logic, and code flow
2. Second subtask should focus on rendering, UI behavior, styling, and presentation

Output the subtasks in JSON format:
{{
    "subtasks": [
        {{
            "description": "subtask 1 description",
            "focus": "implementation"
        }},
        {{
            "description": "subtask 2 description",
            "focus": "rendering"
        }}
    ]
}}"""
    
    plan_response = call_llm_sync(
        plan_handle,
        [{"role": "user", "content": plan_breakdown_prompt}],
        system_prompt=PLAN_AGENT_SYSTEM_PROMPT
    )
    subtasks = parse_subtasks(plan_response)
    print(f"✓ Plan Agent created {len(subtasks)} subtasks")
    for i, st in enumerate(subtasks, 1):
        print(f"  Subtask {i}: {st.get('description', '')[:80]}...")
    
    # Step 2: Parallel exploration
    print("\n[Step 2] Explore Agents: Working in parallel...")
    
    # Prepare exploration requests
    subtask1 = subtasks[0]
    subtask2 = subtasks[1]
    
    explore1_prompt = f"""Explore this coding subtask:

Subtask: {subtask1.get('description', '')}

Code Context:
{code_context}

Focus on: Implementation, structure, logic, data flow, prop handling, state management

Provide your analysis with:
- Summary of findings
- Issues identified
- Recommendations"""
    
    explore2_prompt = f"""Explore this coding subtask:

Subtask: {subtask2.get('description', '')}

Code Context:
{code_context}

Focus on: Rendering, styling, UI behavior, visual feedback, CSS, presentation

Provide your analysis with:
- Summary of findings
- Issues identified
- Recommendations"""
    
    # Call both explore agents in parallel
    explore1_future = explore1_handle.chat.remote(
        ChatCompletionRequest(
            model=llm_config.model_id,
            messages=sanitize_messages_for_request([
                {"role": "system", "content": EXPLORE_AGENT_1_SYSTEM_PROMPT},
                {"role": "user", "content": explore1_prompt}
            ]),
            stream=False,
        )
    )
    
    explore2_future = explore2_handle.chat.remote(
        ChatCompletionRequest(
            model=llm_config.model_id,
            messages=sanitize_messages_for_request([
                {"role": "system", "content": EXPLORE_AGENT_2_SYSTEM_PROMPT},
                {"role": "user", "content": explore2_prompt}
            ]),
            stream=False,
        )
    )
    
    # Wait for both results
    explore1_response = None
    explore2_response = None
    
    # Iterate through generators
    for chunk in explore1_future:
        explore1_response = chunk
    
    for chunk in explore2_future:
        explore2_response = chunk
    
    if explore1_response is None:
        raise RuntimeError("Explore Agent 1 returned no response")
    if explore2_response is None:
        raise RuntimeError("Explore Agent 2 returned no response")
    
    # Convert to dicts
    if hasattr(explore1_response, 'model_dump'):
        explore1_result = explore1_response.model_dump()
    elif hasattr(explore1_response, 'dict'):
        explore1_result = explore1_response.dict()
    else:
        explore1_result = explore1_response
    
    if hasattr(explore2_response, 'model_dump'):
        explore2_result = explore2_response.model_dump()
    elif hasattr(explore2_response, 'dict'):
        explore2_result = explore2_response.dict()
    else:
        explore2_result = explore2_response
    
    explore1_choices = explore1_result.get("choices", [])
    explore2_choices = explore2_result.get("choices", [])
    
    if not explore1_choices:
        raise ValueError("Explore Agent 1 response has no choices")
    if not explore2_choices:
        raise ValueError("Explore Agent 2 response has no choices")
    
    explore1_content = explore1_choices[0].get("message", {}).get("content", "")
    explore2_content = explore2_choices[0].get("message", {}).get("content", "")
    
    if not explore1_content:
        raise ValueError("Explore Agent 1 response has no content")
    if not explore2_content:
        raise ValueError("Explore Agent 2 response has no content")
    
    print(f"✓ Explore Agent 1 completed: {len(explore1_content)} chars")
    print(f"✓ Explore Agent 2 completed: {len(explore2_content)} chars")
    
    # Step 3: Plan Agent synthesizes results
    print("\n[Step 3] Plan Agent: Synthesizing results into fix...")
    
    synthesis_prompt = f"""Synthesize the exploration results into a suggested code fix:

Original Task: {task}

Code Context:
{code_context}

Exploration Results:

--- Explore Agent 1 (Implementation Focus) ---
{explore1_content}

--- Explore Agent 2 (Rendering Focus) ---
{explore2_content}

Based on these findings, provide:
1. Root cause analysis
2. Suggested fix with specific code changes
3. Explanation of why the fix works

Format your response clearly with the fix and explanation."""
    
    synthesis_response = call_llm_sync(
        plan_handle,
        [{"role": "user", "content": synthesis_prompt}],
        system_prompt=PLAN_AGENT_SYSTEM_PROMPT
    )
    
    synthesis_choices = synthesis_response.get("choices", [])
    if not synthesis_choices:
        raise ValueError("Synthesis response has no choices")
    
    synthesis_content = synthesis_choices[0].get("message", {}).get("content", "")
    if not synthesis_content:
        raise ValueError("Synthesis response has no content")
    
    print(f"✓ Synthesis complete: {len(synthesis_content)} chars")
    
    # Return results
    return {
        "id": row_id,
        "task": task,
        "code_context": code_context,
        "subtasks": subtasks,
        "explore1_findings": explore1_content,
        "explore2_findings": explore2_content,
        "suggested_fix": synthesis_content,
        "error": None,
    }


def create_coding_tasks_dataset():
    """Create a synthetic dataset with coding tasks."""
    tasks = [
        {
            "id": 0,
            "task": "Fix the readonly field rendering issue where readonly fields don't show proper styling",
            "code_context": """
function FormField({ name, value, readonly = false }) {
    return (
        <div className="field">
            <label>{name}</label>
            <input 
                type="text" 
                value={value}
                disabled={readonly}
            />
        </div>
    );
}

function Form({ fields }) {
    return fields.map(field => (
        <FormField 
            key={field.id}
            name={field.name}
            value={field.value}
            readonly={field.readonly}
        />
    ));
}
            """
        },
        {
            "id": 1,
            "task": "Fix the async data loading issue where the component renders before data is loaded, causing errors",
            "code_context": """
function DataComponent({ userId }) {
    const [data, setData] = useState(null);
    
    useEffect(() => {
        fetchData(userId).then(setData);
    }, [userId]);
    
    return (
        <div>
            <h1>{data.name}</h1>
            <p>{data.description}</p>
        </div>
    );
}
            """
        },
        {
            "id": 2,
            "task": "Fix the form validation where error messages don't clear when the field is corrected",
            "code_context": """
function ValidatedForm() {
    const [errors, setErrors] = useState({});
    const [values, setValues] = useState({});
    
    const validate = (field, value) => {
        if (!value) {
            setErrors({...errors, [field]: "This field is required"});
        }
    };
    
    return (
        <div>
            <input 
                name="username"
                value={values.username || ''}
                onChange={(e) => {
                    setValues({...values, username: e.target.value});
                    validate("username", e.target.value);
                }}
            />
            {errors.username && <span className="error">{errors.username}</span>}
        </div>
    );
}
            """
        }
    ]
    
    return ray.data.from_items(tasks)


# Main execution
print("=" * 80)
print("Parallel Multi-Agent Coding Assistant")
print("=" * 80)
print("\nDeployments:")
print(f"  - Plan Agent: {PLAN_APP_NAME}")
print(f"  - Explore Agent 1: {EXPLORE1_APP_NAME}")
print(f"  - Explore Agent 2: {EXPLORE2_APP_NAME}")

# Create dataset
print("\nCreating coding tasks dataset...")
ds = create_coding_tasks_dataset()
print(f"Dataset created with {ds.count()} tasks")

# Process all tasks
print("\nProcessing tasks through parallel agent system...")
print("Each task will be:")
print("  1. Broken down by Plan Agent")
print("  2. Explored in parallel by 2 Explore Agents")
print("  3. Synthesized into a fix by Plan Agent")
print()

ds = ds.map(process_coding_task)
ds = ds.materialize()
print("\n✓ All tasks processed!")

# Display results
print("\n" + "=" * 80)
print("FINAL RESULTS")
print("=" * 80)

results = ds.take_all()
for result in results:
    print(f"\n{'='*80}")
    print(f"Task {result.get('id')}: {result.get('task', '')}")
    print(f"{'='*80}")
    
    # Show subtasks
    subtasks = result.get('subtasks', [])
    if subtasks:
        print("\n📋 Subtasks Created:")
        for i, st in enumerate(subtasks, 1):
            print(f"  {i}. {st.get('description', '')}")
    
    # Show exploration results
    explore1 = result.get('explore1_findings', '')
    explore2 = result.get('explore2_findings', '')
    
    if explore1:
        print(f"\n🔍 Explore Agent 1 Findings ({len(explore1)} chars):")
        print("-" * 80)
        print(explore1[:500] + ("..." if len(explore1) > 500 else ""))
    
    if explore2:
        print(f"\n🔍 Explore Agent 2 Findings ({len(explore2)} chars):")
        print("-" * 80)
        print(explore2[:500] + ("..." if len(explore2) > 500 else ""))
    
    # Show suggested fix
    fix = result.get('suggested_fix', '')
    if fix:
        print(f"\n✅ Suggested Fix ({len(fix)} chars):")
        print("-" * 80)
        print(fix)
    
    print()

print("\n" + "=" * 80)
print("All tasks completed!")
print("=" * 80)
