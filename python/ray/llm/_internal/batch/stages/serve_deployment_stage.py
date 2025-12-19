"""The stage that runs serve deployment."""

import asyncio
import logging
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel

from ray import serve
from ray.llm._internal.batch.stages.base import (
    StatefulStage,
    StatefulStageUDF,
)

logger = logging.getLogger(__name__)


def _ensure_json_serializable(obj: Any) -> Any:
    """Recursively convert numpy arrays/scalars to plain Python types.

    Ray Data's Arrow/batch processing can convert Python lists to numpy arrays.
    This function ensures all values are JSON-serializable before sending to
    serve deployments that expect JSON payloads (e.g., vLLM).
    """
    try:
        import numpy as np
    except ImportError:
        np = None

    if np is not None:
        if isinstance(obj, np.ndarray):
            return [_ensure_json_serializable(x) for x in obj.tolist()]
        if isinstance(obj, np.generic):
            return obj.item()

    if isinstance(obj, dict):
        return {k: _ensure_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_ensure_json_serializable(v) for v in obj]
    return obj


def _consume_iterator_to_list(obj: Any) -> Any:
    """Recursively consume any iterator-like objects into lists.

    This handles ValidatorIterator from Pydantic which can't be pickled.
    """
    # Check for ValidatorIterator by name (avoids importing pydantic_core)
    type_name = type(obj).__name__
    if "Iterator" in type_name or "iterator" in type_name:
        try:
            return [_consume_iterator_to_list(item) for item in obj]
        except (TypeError, StopIteration):
            return obj

    if isinstance(obj, dict):
        return {k: _consume_iterator_to_list(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_consume_iterator_to_list(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(_consume_iterator_to_list(item) for item in obj)

    return obj


def _sanitize_messages_dict(messages: List[Any]) -> List[Dict[str, Any]]:
    """Sanitize a list of message dicts to ensure tool_calls are plain lists.

    Args:
        messages: List of message dicts that may contain ValidatorIterator.

    Returns:
        List of sanitized message dicts with tool_calls as plain lists.
    """
    import json

    sanitized_messages = []
    for msg in messages:
        if isinstance(msg, dict):
            msg_copy = msg.copy()
            # Handle tool_calls field
            if "tool_calls" in msg_copy and msg_copy["tool_calls"] is not None:
                tool_calls = msg_copy["tool_calls"]
                # Force conversion to list if it's an iterator or other iterable
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
                    # Consume any nested iterators
                    tc = _consume_iterator_to_list(tc)
                    if isinstance(tc, dict):
                        # JSON round-trip to ensure all nested values are plain
                        try:
                            tc = json.loads(json.dumps(tc))
                        except (TypeError, ValueError):
                            pass
                    sanitized_tool_calls.append(tc)
                msg_copy["tool_calls"] = sanitized_tool_calls
            sanitized_messages.append(msg_copy)
        else:
            sanitized_messages.append(msg)
    return sanitized_messages


def _sanitize_request_tool_calls(request_obj: Any, dtype: Type[Any]) -> Any:
    """Sanitize ChatCompletionRequest to fix Pydantic ValidatorIterator serialization.

    This addresses a known Pydantic bug where tool_calls fields become ValidatorIterator
    objects that cannot be pickled for Ray remote calls.

    References:
    - vLLM PR that introduces the workaround: https://github.com/vllm-project/vllm/pull/9951
    - Pydantic Issue: https://github.com/pydantic/pydantic/issues/9467

    Args:
        request_obj: The ChatCompletionRequest object to sanitize.
        dtype: The dtype class (ChatCompletionRequest) to use for reconstruction.

    Returns:
        A sanitized ChatCompletionRequest with tool_calls as plain lists.
    """
    import json

    # Preserve original tools (they're already Pydantic models)
    original_tools = (
        request_obj.tools
        if hasattr(request_obj, "tools") and request_obj.tools
        else None
    )

    # Convert request to dict, handling ValidatorIterator by consuming iterables
    # Use model_dump with mode='python' which should convert iterators to lists
    request_dict = request_obj.model_dump(mode="python")

    # Consume any iterators that model_dump didn't convert
    request_dict = _consume_iterator_to_list(request_dict)

    # Sanitize messages to ensure tool_calls are concrete lists
    if "messages" in request_dict and request_dict["messages"]:
        request_dict["messages"] = _sanitize_messages_dict(request_dict["messages"])

    # JSON round-trip to ensure everything is plain Python types
    # (except for tools which we'll restore after)
    request_dict_no_tools = {k: v for k, v in request_dict.items() if k != "tools"}
    try:
        request_dict_no_tools = json.loads(json.dumps(request_dict_no_tools))
    except (TypeError, ValueError):
        # If JSON serialization fails, try consuming iterators again
        request_dict_no_tools = _consume_iterator_to_list(request_dict_no_tools)

    # Restore original tools (Pydantic models) if they exist
    if original_tools is not None:
        request_dict_no_tools["tools"] = original_tools
    elif "tools" in request_dict:
        # Keep tools from dict if no original (might be plain dicts)
        request_dict_no_tools["tools"] = request_dict["tools"]

    # Recreate using model_construct to bypass validation
    # This prevents ValidatorIterator from being created
    request_obj = dtype.model_construct(**request_dict_no_tools)

    # Also call vLLM's sanitization if available
    try:
        from ray.llm._internal.serve.core.ingress.ingress import (
            _sanitize_chat_completion_request,
        )

        request_obj = _sanitize_chat_completion_request(request_obj)
    except (ImportError, TypeError, AttributeError):
        pass

    return request_obj


class ServeDeploymentStageUDF(StatefulStageUDF):
    def __init__(
        self,
        data_column: str,
        expected_input_keys: List[str],
        *,
        deployment_name: str,
        app_name: str,
        dtype_mapping: Dict[str, Type[Any]],
    ):
        """
        Initialize the ServeDeploymentStageUDF.

        Args:
            data_column: The data column name.
            expected_input_keys: The expected input keys of the stage.
            deployment_name: The name of the deployment.
            app_name: The name of the deployment app.
            dtype_mapping: The mapping of the request class name to the request class.
        """
        super().__init__(data_column, expected_input_keys)
        self._dtype_mapping = dtype_mapping

        # Using stream=True as LLM serve deployments return async generators.
        # TODO (Kourosh): Generalize this to support non-streaming deployments.
        self._dh = serve.get_deployment_handle(deployment_name, app_name).options(
            stream=True
        )
        self.request_id = 0

    def _prepare_request(
        self, row: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Optional[Type[Any]], str]:
        """
        Decorate the request with metadata related to the batch.

        Args:
            row: The row.

        Returns:
            A tuple of (decorated_request, dtype, method_name). dtype is the class of the request object and
            can be None if the serve deployment accepts a raw dict. method_name is the name of the method to
            invoke on the serve deployment.
        """
        method = row.get("method")
        dtype_name = row.get("dtype")

        dtype = None
        if dtype_name is not None:
            if not self._dtype_mapping or dtype_name not in self._dtype_mapping:
                raise ValueError(
                    f"{dtype_name} must be provided in ServeDeploymentProcessorConfig's dtype_mapping."
                )
            dtype = self._dtype_mapping[dtype_name]

        request_kwargs = row.pop("request_kwargs")
        # Sanitize request_kwargs to ensure all values are JSON-serializable.
        # Ray Data may convert Python lists to numpy arrays during batch processing.
        request_kwargs = _ensure_json_serializable(request_kwargs)

        # Pre-sanitize messages to prevent ValidatorIterator from being created.
        # This handles the case where messages contain tool_calls from a previous turn.
        if "messages" in request_kwargs and request_kwargs["messages"]:
            request_kwargs["messages"] = _sanitize_messages_dict(
                request_kwargs["messages"]
            )

        request = {
            "request_id": str(self.request_id),
            "idx_in_batch": row[self.IDX_IN_BATCH_COLUMN],
            **request_kwargs,
        }
        self.request_id += 1

        return request, dtype, method

    async def generate_async(
        self, row: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any], float]:
        """
        Run the serve deployment.

        Args:
            row: The row to run the serve deployment on.

        Returns:
            The response from the serve deployment.
        """
        request, dtype, method = self._prepare_request(row)
        
        # Create the request object
        request_obj = dtype(**request) if dtype else request

        # Sanitize ChatCompletionRequest after creation to fix ValidatorIterator serialization.
        # This addresses a known Pydantic bug where tool_calls fields become ValidatorIterator
        # objects that cannot be pickled for Ray remote calls.
        # See: https://github.com/pydantic/pydantic/issues/9467
        # This mimics the approach used in ray.llm._internal.serve.core.ingress.ingress
        if dtype is not None:
            try:
                from ray.serve.llm.openai_api_models import ChatCompletionRequest
                if isinstance(request_obj, ChatCompletionRequest):
                    request_obj = _sanitize_request_tool_calls(request_obj, dtype)
            except (ImportError, TypeError, AttributeError):
                # If sanitization fails, continue anyway
                pass

        if getattr(self._dh, method) is None:
            raise ValueError(f"Method {method} not found in the serve deployment.")

        # Final sanitization right before serialization
        # This ensures tool_calls is always a concrete list, not an iterable/ValidatorIterator
        # Workaround from: https://github.com/pydantic/pydantic/issues/9467
        if dtype is not None:
            try:
                from ray.serve.llm.openai_api_models import ChatCompletionRequest
                if isinstance(request_obj, ChatCompletionRequest):
                    request_obj = _sanitize_request_tool_calls(request_obj, dtype)
            except (ImportError, TypeError, AttributeError, ValueError):
                pass

        t = time.perf_counter()
        # Directly using anext() requires python3.10 and above
        output_data = await getattr(self._dh, method).remote(request_obj).__anext__()
        time_taken = time.perf_counter() - t

        # Convert the output data to a dict if it is a Pydantic model.
        if isinstance(output_data, BaseModel):
            output_data = output_data.model_dump()

        return request, output_data, time_taken

    async def udf(self, batch: List[Dict[str, Any]]) -> AsyncIterator[Dict[str, Any]]:
        """
        Run the serve deployment.

        Args:
            batch: A list of rows to run the serve deployment on.

        Yields:
            Dict[str, Any]: A dictionary containing the response from the serve deployment
            along with processing metadata.
        """
        batch_uuid = uuid.uuid4()
        t = time.perf_counter()
        tasks = [asyncio.create_task(self.generate_async(row)) for row in batch]

        for resp in asyncio.as_completed(tasks):
            request, output, time_taken = await resp

            yield {
                "request_id": request["request_id"],
                self.IDX_IN_BATCH_COLUMN: request["idx_in_batch"],
                "batch_uuid": batch_uuid.hex,
                "time_taken": time_taken,
                **output,
            }

        batch_time_taken = time.perf_counter() - t
        logger.info(
            "[LLM Batch - Serve Deployment] Elapsed time for batch %s with size %d: %s",
            batch_uuid.hex,
            len(batch),
            batch_time_taken,
        )


class ServeDeploymentStage(StatefulStage):
    fn: Type[StatefulStageUDF] = ServeDeploymentStageUDF

    def get_required_input_keys(self) -> Dict[str, str]:
        return {
            "method": "Name of the method to invoke on the serve deployment.",
            "request_kwargs": "The request_kwargs to construct the request to the serve deployment.",
        }
