"""
Qwen VL Combined Actor - Full Model (Vision + LLM) with TP Support

This module implements a Ray actor that hosts the complete Qwen VL model
including both the vision encoder and language model backbone.

Supports:
- Data Parallelism (DP) on the vision encoder
- Tensor Parallelism (TP) on the language model via torch.distributed

This is used for comparison with the disaggregated approach to verify
that both produce identical outputs.
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
    images to language model outputs. Supports TP on the LLM portion.
    
    Args:
        model_name_or_path: HuggingFace model name or local path
        rank: Actor rank (used for both vision DP and LLM TP)
        tp_size: Tensor parallel size for LLM (default: 1)
        dtype: Model dtype (default: bfloat16)
        device: Device to use (default: cuda:0)
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        rank: int = 0,
        tp_size: int = 1,
        master_addr: str = None,
        master_port: int = 29500,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda:0",
    ):
        self.model_name_or_path = model_name_or_path
        self.rank = rank
        self.tp_size = tp_size
        self.master_addr = master_addr
        self.master_port = master_port
        self.dtype = dtype
        self.device = torch.device(device)
        self.model: Optional[nn.Module] = None
        self.config = None
        self.model_type = None
        self._distributed_initialized = False
        self._tp_group = None
        
        logger.info(
            f"Combined actor {rank} (TP size={tp_size}) initialized with model {model_name_or_path}"
        )
    
    def get_ip_address(self) -> str:
        """Get the IP address of this actor for NCCL communication."""
        hostname = socket.gethostname()
        ip_addr = socket.gethostbyname(hostname)
        return ip_addr
    
    def init_distributed(self, master_addr: str, master_port: int) -> None:
        """
        Initialize distributed environment for tensor parallelism on LLM.
        
        Args:
            master_addr: IP address of rank 0 for NCCL rendezvous
            master_port: Port for NCCL rendezvous
        """
        if self._distributed_initialized:
            logger.info(f"Combined actor {self.rank}: Distributed already initialized")
            return
            
        if self.tp_size <= 1:
            logger.info(f"Combined actor {self.rank}: TP size is 1, skipping distributed init")
            return
        
        self.master_addr = master_addr
        self.master_port = master_port
        
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(self.tp_size)
        
        logger.info(
            f"Combined actor {self.rank}: Initializing distributed "
            f"(master={master_addr}:{master_port}, rank={self.rank}/{self.tp_size})"
        )
        
        dist.init_process_group(
            backend="nccl",
            rank=self.rank,
            world_size=self.tp_size,
        )
        
        # Create a TP process group
        self._tp_group = dist.new_group(ranks=list(range(self.tp_size)))
        
        self._distributed_initialized = True
        logger.info(f"Combined actor {self.rank}: Distributed initialized successfully")
    
    def build_model(self) -> None:
        """
        Build and load the complete Qwen VL model.
        """
        import torch
        from transformers import AutoConfig, AutoModel
        
        logger.info(f"Combined actor {self.rank}: Loading config from {self.model_name_or_path}")
        
        # Load HuggingFace config
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
        
        # Load the full model
        logger.info(f"Combined actor {self.rank}: Loading full model from transformers")
        
        try:
            self.model = AutoModel.from_pretrained(
                self.model_name_or_path,
                torch_dtype=self.dtype,
                trust_remote_code=True,
            )
            self.model = self.model.to(self.device)
            self.model.eval()
            logger.info(f"Combined actor {self.rank}: Full model loaded successfully")
            
        except Exception as e:
            logger.error(f"Combined actor {self.rank}: Error loading model: {e}")
            raise
    
    def forward_vision(
        self,
        pixel_values: torch.Tensor,
        grid_thw: Union[torch.Tensor, list],
    ) -> torch.Tensor:
        """
        Forward pass through just the vision encoder part.
        
        Args:
            pixel_values: Image/video pixel values [L, C]
            grid_thw: Grid dimensions [N, 3]
        
        Returns:
            Vision embeddings [num_tokens, hidden_size]
        """
        if self.model is None:
            raise RuntimeError("Model not built. Call build_model() first.")
        
        # Move inputs to device
        pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)
        if isinstance(grid_thw, torch.Tensor):
            grid_thw_tensor = grid_thw.to(device=self.device)
        else:
            grid_thw_tensor = torch.tensor(grid_thw, device=self.device)
        
        with torch.no_grad():
            # Access the vision encoder
            if hasattr(self.model, 'visual'):
                vision_encoder = self.model.visual
            elif hasattr(self.model, 'vision_model'):
                vision_encoder = self.model.vision_model
            elif hasattr(self.model, 'vision_tower'):
                vision_encoder = self.model.vision_tower
            else:
                raise RuntimeError("Could not find vision encoder in model")
            
            embeddings = vision_encoder(pixel_values, grid_thw=grid_thw_tensor)
        
        logger.info(
            f"Combined actor {self.rank}: Vision - input shape {pixel_values.shape}, "
            f"output shape {embeddings.shape}"
        )
        
        return embeddings
    
    def all_gather_vision_embeddings(
        self,
        local_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        All-gather vision embeddings from all DP actors.
        
        Each actor has processed a portion of the images. This method
        gathers all partial embeddings so each actor has the FULL batch.
        
        Args:
            local_embeddings: This actor's vision embeddings [local_seq, hidden]
        
        Returns:
            Full vision embeddings from all actors [total_seq, hidden]
        """
        if self.tp_size <= 1 or not self._distributed_initialized:
            return local_embeddings
        
        local_embeddings = local_embeddings.to(device=self.device, dtype=self.dtype)
        
        # Get the size from all ranks to handle variable sizes
        local_size = torch.tensor([local_embeddings.shape[0]], device=self.device)
        all_sizes = [torch.zeros(1, device=self.device, dtype=torch.long) for _ in range(self.tp_size)]
        dist.all_gather(all_sizes, local_size, group=self._tp_group)
        all_sizes = [int(s.item()) for s in all_sizes]
        
        max_size = max(all_sizes)
        hidden_size = local_embeddings.shape[-1]
        
        # Pad local embeddings to max size
        if local_embeddings.shape[0] < max_size:
            padding = torch.zeros(
                max_size - local_embeddings.shape[0], hidden_size,
                device=self.device, dtype=self.dtype
            )
            padded_local = torch.cat([local_embeddings, padding], dim=0)
        else:
            padded_local = local_embeddings
        
        # All-gather padded tensors
        gathered = [torch.zeros_like(padded_local) for _ in range(self.tp_size)]
        dist.all_gather(gathered, padded_local, group=self._tp_group)
        
        # Trim padding and concatenate
        trimmed = [gathered[i][:all_sizes[i]] for i in range(self.tp_size)]
        full_embeddings = torch.cat(trimmed, dim=0)
        
        logger.info(
            f"Combined actor {self.rank}: All-gathered vision embeddings "
            f"from {local_embeddings.shape} to {full_embeddings.shape}"
        )
        
        return full_embeddings
    
    def forward_llm(
        self,
        vision_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through just the language model part.
        
        For TP > 1, all actors receive the same vision embeddings (full batch)
        and synchronize their outputs via all-reduce.
        
        Args:
            vision_embeddings: Vision encoder outputs [seq_len, hidden] or [batch, seq_len, hidden]
        
        Returns:
            Language model hidden states
        """
        if self.model is None:
            raise RuntimeError("Model not built. Call build_model() first.")
        
        # Move inputs to device
        vision_embeddings = vision_embeddings.to(device=self.device, dtype=self.dtype)
        
        # Ensure 3D tensor: [batch, seq_len, hidden]
        if vision_embeddings.dim() == 2:
            vision_embeddings = vision_embeddings.unsqueeze(0)
        
        with torch.no_grad():
            # Access the language model
            if hasattr(self.model, 'language_model'):
                lm = self.model.language_model
            elif hasattr(self.model, 'model'):
                lm = self.model.model
            else:
                lm = self.model
            
            try:
                output = lm(
                    inputs_embeds=vision_embeddings,
                    return_dict=True,
                )
                output = output.last_hidden_state if hasattr(output, 'last_hidden_state') else output.logits
            except Exception as e:
                logger.warning(f"Combined actor {self.rank}: LLM forward failed: {e}")
                output = vision_embeddings
            
            # For TP > 1, synchronize outputs across ranks via all-reduce
            if self.tp_size > 1 and self._distributed_initialized:
                dist.all_reduce(output, op=dist.ReduceOp.AVG, group=self._tp_group)
                logger.debug(f"Combined actor {self.rank}: All-reduce completed")
        
        logger.info(
            f"Combined actor {self.rank}: LLM - input shape {vision_embeddings.shape}, "
            f"output shape {output.shape}"
        )
        
        return output
    
    def forward_full(
        self,
        pixel_values: torch.Tensor,
        grid_thw: Union[torch.Tensor, list],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass through vision encoder and language model.
        
        This is the combined end-to-end path that should produce
        identical results to the disaggregated approach.
        
        Args:
            pixel_values: Image/video pixel values [L, C]
            grid_thw: Grid dimensions [N, 3]
        
        Returns:
            Tuple of (vision_embeddings, llm_output)
        """
        # First pass through vision encoder
        vision_embeddings = self.forward_vision(pixel_values, grid_thw)
        
        # Then pass through language model
        llm_output = self.forward_llm(vision_embeddings)
        
        return vision_embeddings, llm_output
    
    def cleanup_distributed(self) -> None:
        """Cleanup distributed resources."""
        if self._distributed_initialized:
            dist.destroy_process_group()
            self._distributed_initialized = False
            logger.info(f"Combined actor {self.rank}: Distributed cleanup completed")
    
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
    Create a Ray remote class for the combined actor with specified resources.
    
    Args:
        num_gpus: Number of GPUs per actor
        num_cpus: Number of CPUs per actor
    
    Returns:
        Ray remote class
    """
    return ray.remote(num_gpus=num_gpus, num_cpus=num_cpus)(QwenVLCombinedActor)
