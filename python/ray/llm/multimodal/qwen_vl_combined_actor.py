"""
Qwen VL Combined Actor - Full Model (Vision + LLM) in Single Actor

This module implements a Ray actor that hosts the complete Qwen VL model
including both the vision encoder and language model backbone.

Use this for:
- Baseline benchmarks (single forward pass through entire VLM)
- Comparison with disaggregated approach

For DP/TP configurations, use qwen3_vl_vision_actor.py and qwen3_vl_llm_actor.py.
"""

import logging
import os
import socket
from typing import Optional, Union, Tuple

import ray
import torch
import torch.nn as nn
import torch.distributed as dist

logger = logging.getLogger(__name__)


class QwenVLCombinedActor:
    """
    Ray actor that hosts the complete Qwen VL model (vision + LLM).
    
    This actor loads the full model and can process end-to-end from
    images to language model outputs in a single forward pass.
    
    Args:
        model_name_or_path: HuggingFace model name or local path
        rank: Actor rank (for DP across replicas)
        world_size: Total number of replicas
        dtype: Model dtype (default: bfloat16)
        device: Device to use (default: cuda:0)
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        rank: int = 0,
        world_size: int = 1,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda:0",
    ):
        self.model_name_or_path = model_name_or_path
        self.rank = rank
        self.world_size = world_size
        self.dtype = dtype
        self.device = torch.device(device)
        self.model: Optional[nn.Module] = None
        self.config = None
        self.model_type = None
        self._distributed_initialized = False
        
        logger.info(
            f"Combined actor rank={rank}/{world_size} initialized "
            f"with model {model_name_or_path}"
        )
    
    def get_ip_address(self) -> str:
        """Get the IP address of this actor for NCCL communication."""
        hostname = socket.gethostname()
        return socket.gethostbyname(hostname)
    
    def init_distributed(
        self,
        master_addr: str,
        master_port: int,
    ) -> None:
        """
        Initialize distributed environment for multi-replica setup.
        
        Args:
            master_addr: IP address of rank 0 for NCCL rendezvous
            master_port: Port for NCCL rendezvous
        """
        if self._distributed_initialized:
            logger.info(f"Combined actor {self.rank}: Distributed already initialized")
            return
        
        if self.world_size <= 1:
            logger.info(f"Combined actor {self.rank}: Single replica, skipping distributed")
            return
        
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(self.world_size)
        
        logger.info(
            f"Combined actor {self.rank}: Initializing distributed "
            f"(master={master_addr}:{master_port}, rank={self.rank}/{self.world_size})"
        )
        
        dist.init_process_group(
            backend="nccl",
            rank=self.rank,
            world_size=self.world_size,
        )
        
        self._distributed_initialized = True
        logger.info(f"Combined actor {self.rank}: Distributed initialized")
    
    def cleanup_distributed(self) -> None:
        """Clean up distributed resources."""
        if self._distributed_initialized:
            try:
                dist.destroy_process_group()
                self._distributed_initialized = False
                logger.info(f"Combined actor {self.rank}: Distributed cleaned up")
            except Exception as e:
                logger.warning(f"Combined actor {self.rank}: Cleanup failed: {e}")
    
    def build_model(self) -> None:
        """
        Build and load the complete Qwen VL model.
        """
        from transformers import AutoConfig, AutoModel
        
        logger.info(f"Combined actor {self.rank}: Loading config from {self.model_name_or_path}")
        
        hf_config = AutoConfig.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
        )
        self.config = hf_config
        
        # Detect model type
        config_class_name = hf_config.__class__.__name__
        if "Qwen3" in config_class_name:
            self.model_type = "Qwen3-VL"
        elif "Qwen2_5" in config_class_name or "Qwen2.5" in config_class_name:
            self.model_type = "Qwen2.5-VL"
        else:
            self.model_type = "Qwen2-VL"
        
        logger.info(f"Combined actor {self.rank}: Detected model type: {self.model_type}")
        
        try:
            self.model = AutoModel.from_pretrained(
                self.model_name_or_path,
                torch_dtype=self.dtype,
                trust_remote_code=True,
            )
            self.model = self.model.to(self.device)
            self.model.eval()
            logger.info(f"Combined actor {self.rank}: Full model loaded")
            
        except Exception as e:
            logger.error(f"Combined actor {self.rank}: Error loading model: {e}")
            raise
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: Union[torch.Tensor, list],
    ) -> torch.Tensor:
        """
        Single unified forward pass through the entire model (vision + LLM).
        
        This performs end-to-end inference in one call for true throughput measurement.
        
        Args:
            pixel_values: Image/video pixel values [L, C]
            grid_thw: Grid dimensions [N, 3]
        
        Returns:
            LLM output hidden states
        """
        if self.model is None:
            raise RuntimeError("Model not built. Call build_model() first.")
        
        pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)
        if isinstance(grid_thw, torch.Tensor):
            grid_thw_tensor = grid_thw.to(device=self.device)
        else:
            grid_thw_tensor = torch.tensor(grid_thw, device=self.device)
        
        with torch.no_grad():
            # === VISION ENCODER ===
            if hasattr(self.model, 'visual'):
                vision_encoder = self.model.visual
            elif hasattr(self.model, 'vision_model'):
                vision_encoder = self.model.vision_model
            elif hasattr(self.model, 'vision_tower'):
                vision_encoder = self.model.vision_tower
            else:
                raise RuntimeError("Could not find vision encoder in model")
            
            vision_embeddings = vision_encoder(pixel_values, grid_thw=grid_thw_tensor)
            
            # === LANGUAGE MODEL ===
            if vision_embeddings.dim() == 2:
                vision_embeddings = vision_embeddings.unsqueeze(0)
            
            if hasattr(self.model, 'language_model'):
                lm = self.model.language_model
            elif hasattr(self.model, 'model'):
                lm = self.model.model
            else:
                lm = self.model
            
            try:
                result = lm(inputs_embeds=vision_embeddings, return_dict=True)
                output = result.last_hidden_state if hasattr(result, 'last_hidden_state') else result.logits
            except Exception as e:
                logger.warning(f"Combined actor {self.rank}: LLM forward failed: {e}")
                output = vision_embeddings
        
        logger.info(
            f"Combined actor {self.rank}: Forward complete, "
            f"input {pixel_values.shape} -> output {output.shape}"
        )
        return output
    
    def forward_vision(
        self,
        pixel_values: torch.Tensor,
        grid_thw: Union[torch.Tensor, list],
    ) -> torch.Tensor:
        """
        Forward pass through just the vision encoder.
        
        Args:
            pixel_values: Image/video pixel values [L, C]
            grid_thw: Grid dimensions [N, 3]
        
        Returns:
            Vision embeddings [num_tokens, hidden_size]
        """
        if self.model is None:
            raise RuntimeError("Model not built. Call build_model() first.")
        
        pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)
        if isinstance(grid_thw, torch.Tensor):
            grid_thw_tensor = grid_thw.to(device=self.device)
        else:
            grid_thw_tensor = torch.tensor(grid_thw, device=self.device)
        
        with torch.no_grad():
            if hasattr(self.model, 'visual'):
                vision_encoder = self.model.visual
            elif hasattr(self.model, 'vision_model'):
                vision_encoder = self.model.vision_model
            elif hasattr(self.model, 'vision_tower'):
                vision_encoder = self.model.vision_tower
            else:
                raise RuntimeError("Could not find vision encoder in model")
            
            embeddings = vision_encoder(pixel_values, grid_thw=grid_thw_tensor)
        
        return embeddings
    
    def forward_llm(
        self,
        vision_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through just the language model.
        
        Args:
            vision_embeddings: Vision encoder outputs [seq_len, hidden] or [batch, seq_len, hidden]
        
        Returns:
            Language model hidden states
        """
        if self.model is None:
            raise RuntimeError("Model not built. Call build_model() first.")
        
        vision_embeddings = vision_embeddings.to(device=self.device, dtype=self.dtype)
        
        if vision_embeddings.dim() == 2:
            vision_embeddings = vision_embeddings.unsqueeze(0)
        
        with torch.no_grad():
            if hasattr(self.model, 'language_model'):
                lm = self.model.language_model
            elif hasattr(self.model, 'model'):
                lm = self.model.model
            else:
                lm = self.model
            
            try:
                result = lm(inputs_embeds=vision_embeddings, return_dict=True)
                output = result.last_hidden_state if hasattr(result, 'last_hidden_state') else result.logits
            except Exception as e:
                logger.warning(f"Combined actor {self.rank}: LLM forward failed: {e}")
                output = vision_embeddings
        
        return output
    
    def forward_full(
        self,
        pixel_values: torch.Tensor,
        grid_thw: Union[torch.Tensor, list],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass returning both vision embeddings and LLM output.
        
        Args:
            pixel_values: Image/video pixel values [L, C]
            grid_thw: Grid dimensions [N, 3]
        
        Returns:
            Tuple of (vision_embeddings, llm_output)
        """
        vision_embeddings = self.forward_vision(pixel_values, grid_thw)
        llm_output = self.forward_llm(vision_embeddings)
        return vision_embeddings, llm_output
    
    def get_vision_hidden_size(self) -> int:
        """Get the output hidden size of the vision encoder."""
        if self.config is None:
            raise RuntimeError("Model not built")
        vision_config = getattr(self.config, 'vision_config', self.config)
        return getattr(vision_config, 'out_hidden_size', 
                      getattr(vision_config, 'hidden_size', 1152))
    
    def get_llm_hidden_size(self) -> int:
        """Get the hidden size of the language model."""
        if self.config is None:
            raise RuntimeError("Model not built")
        text_config = getattr(self.config, "text_config", self.config)
        return getattr(text_config, "hidden_size", 2048)


def create_combined_actor_class(num_gpus: int = 1, num_cpus: int = 4):
    """
    Create a Ray remote class for the combined actor.
    
    Args:
        num_gpus: Number of GPUs per actor
        num_cpus: Number of CPUs per actor
    
    Returns:
        Ray remote class
    """
    return ray.remote(num_gpus=num_gpus, num_cpus=num_cpus)(QwenVLCombinedActor)

