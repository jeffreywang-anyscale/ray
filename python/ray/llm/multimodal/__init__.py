"""
Ray LLM Multimodal Module

This module provides distributed multimodal inference with:
- Vision encoders with configurable Data Parallelism (DP)
- Language models with Tensor Parallelism (TP)

The vision and language components can scale independently:
- Increase vision DP actors for higher image throughput
- Increase LLM TP size for larger language models

Example:
    from ray.llm.multimodal import Qwen3VLDistributedModel
    
    model = Qwen3VLDistributedModel(
        model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct",
        num_vision_actors=4,  # 4 GPUs for vision DP
        tp_size=2,            # 2 GPUs for LLM TP
    )
    model.setup()
    
    output = model.forward(pixel_values_list, grid_thw_list)
"""

from ray.llm.multimodal.qwen3_vl_vision_actor import (
    Qwen3VLVisionActor,
    create_vision_actor_class,
)
from ray.llm.multimodal.qwen3_vl_llm_actor import (
    Qwen3VLLLMActor,
    create_llm_actor_class,
)
from ray.llm.multimodal.qwen3_vl_forward import (
    Qwen3VLDistributedModel,
    create_dummy_batch,
)
from ray.llm.multimodal.actor_group import ActorGroup

__all__ = [
    # Vision encoder
    "Qwen3VLVisionActor",
    "create_vision_actor_class",
    # Language model
    "Qwen3VLLLMActor", 
    "create_llm_actor_class",
    # Distributed model
    "Qwen3VLDistributedModel",
    "create_dummy_batch",
    # Utilities
    "ActorGroup",
]

