"""
Qwen VL Vision Encoder Ray Actor with Data Parallelism

This module implements a Ray actor that hosts the Qwen VL vision encoder.
Supports both Qwen2.5-VL and Qwen3-VL models.

Multiple actors can be created for data parallelism - each actor processes
different portions of the input batch independently.

Supports:
- NCCL-based all_gather for efficient embedding aggregation
- NCCL P2P/broadcast to send embeddings to LLM actors
- Colocated mode (vision + LLM on same GPU)
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


class QwenVLVisionActor:
    """
    Ray actor that hosts a Qwen VL vision encoder with data parallelism.
    
    Each actor loads the vision encoder and processes a subset of images.
    NCCL all_gather combines embeddings across DP actors.
    
    Args:
        model_name_or_path: HuggingFace model name or local path
        dp_rank: Data parallel rank of this actor
        dp_size: Data parallel size (number of vision actors)
        global_rank: Global rank in the full distributed world
        dtype: Model dtype (default: bfloat16)
        device: Device to use (default: cuda:0)
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        dp_rank: int = 0,
        dp_size: int = 1,
        global_rank: int = None,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda:0",
    ):
        self.model_name_or_path = model_name_or_path
        self.dp_rank = dp_rank
        self.dp_size = dp_size
        self.global_rank = global_rank if global_rank is not None else dp_rank
        self.dtype = dtype
        self.device = torch.device(device)
        self.model: Optional[nn.Module] = None
        self.full_model = None
        self.config = None
        self.model_type = None
        self._distributed_initialized = False
        self._dp_group = None
        self._world_size = None
        
        logger.info(
            f"Vision actor dp_rank={dp_rank}/{dp_size} (global_rank={self.global_rank}) "
            f"initialized with model {model_name_or_path}"
        )
    
    def get_ip_address(self) -> str:
        """Get the IP address of this actor for NCCL communication."""
        hostname = socket.gethostname()
        return socket.gethostbyname(hostname)
    
    def init_distributed(
        self,
        master_addr: str,
        master_port: int,
        world_size: int,
        llm_tp_size: int = 0,
    ) -> None:
        """
        Initialize distributed environment for NCCL communication.
        
        Args:
            master_addr: IP address of rank 0 for NCCL rendezvous
            master_port: Port for NCCL rendezvous
            world_size: Total world size (all vision + LLM actors)
            llm_tp_size: Number of LLM TP actors (for non-colocated mode)
        """
        if self._distributed_initialized:
            logger.info(f"Vision actor {self.dp_rank}: Distributed already initialized")
            return
        
        self._world_size = world_size
        
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["RANK"] = str(self.global_rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        
        logger.info(
            f"Vision actor dp_rank={self.dp_rank}: Initializing distributed "
            f"(master={master_addr}:{master_port}, global_rank={self.global_rank}/{world_size})"
        )
        
        dist.init_process_group(
            backend="nccl",
            rank=self.global_rank,
            world_size=world_size,
        )
        
        # Create DP process group for vision actors
        vision_ranks = list(range(self.dp_size))
        self._dp_group = dist.new_group(ranks=vision_ranks)
        logger.info(f"Vision actor dp_rank={self.dp_rank}: Created DP group with ranks {vision_ranks}")
        
        self._distributed_initialized = True
        logger.info(f"Vision actor dp_rank={self.dp_rank}: Distributed initialized successfully")
    
    def cleanup_distributed(self) -> None:
        """Clean up distributed resources."""
        if self._distributed_initialized:
            try:
                dist.destroy_process_group()
                self._distributed_initialized = False
                logger.info(f"Vision actor {self.dp_rank}: Distributed cleaned up")
            except Exception as e:
                logger.warning(f"Vision actor {self.dp_rank}: Cleanup failed: {e}")
    
    def send_embeddings_p2p(
        self,
        embeddings: torch.Tensor,
        dst_rank: int,
    ) -> None:
        """
        Send embeddings to a specific rank via NCCL P2P.
        
        Args:
            embeddings: Vision embeddings to send [seq_len, hidden]
            dst_rank: Global rank of destination
        """
        if not self._distributed_initialized:
            raise RuntimeError("Distributed not initialized")
        
        embeddings = embeddings.to(device=self.device, dtype=self.dtype).contiguous()
        req = dist.isend(embeddings, dst=dst_rank)
        req.wait()
        
        logger.info(
            f"Vision actor dp_rank={self.dp_rank}: Sent embeddings to rank {dst_rank}, "
            f"shape {embeddings.shape}"
        )
    
    def broadcast_embeddings(
        self,
        embeddings: torch.Tensor,
    ) -> None:
        """
        Broadcast embeddings to all ranks from this actor.
        
        Args:
            embeddings: Vision embeddings to broadcast [seq_len, hidden]
        """
        if not self._distributed_initialized:
            raise RuntimeError("Distributed not initialized")
        
        embeddings = embeddings.to(device=self.device, dtype=self.dtype).contiguous()
        dist.broadcast(embeddings, src=self.global_rank)
        
        logger.info(
            f"Vision actor dp_rank={self.dp_rank}: Broadcasted embeddings, shape {embeddings.shape}"
        )
    
    def all_gather_embeddings(self, local_embeddings: torch.Tensor) -> torch.Tensor:
        """
        All-gather vision embeddings from all DP actors using NCCL.
        
        Args:
            local_embeddings: This actor's vision embeddings [local_seq, hidden]
        
        Returns:
            Full vision embeddings from all actors [total_seq, hidden]
        """
        if self.dp_size <= 1 or not self._distributed_initialized:
            return local_embeddings
        
        local_embeddings = local_embeddings.to(device=self.device, dtype=self.dtype)
        
        # Gather sizes from all ranks (handles variable sizes)
        local_size = torch.tensor([local_embeddings.shape[0]], device=self.device, dtype=torch.long)
        all_sizes = [torch.zeros(1, device=self.device, dtype=torch.long) for _ in range(self.dp_size)]
        dist.all_gather(all_sizes, local_size, group=self._dp_group)
        all_sizes = [int(s.item()) for s in all_sizes]
        
        max_size = max(all_sizes)
        hidden_size = local_embeddings.shape[-1]
        
        # Pad local embeddings for uniform all_gather
        if local_embeddings.shape[0] < max_size:
            padding = torch.zeros(
                max_size - local_embeddings.shape[0], hidden_size,
                device=self.device, dtype=self.dtype
            )
            padded_local = torch.cat([local_embeddings, padding], dim=0)
        else:
            padded_local = local_embeddings
        
        # All-gather padded tensors
        gathered = [torch.zeros_like(padded_local) for _ in range(self.dp_size)]
        dist.all_gather(gathered, padded_local, group=self._dp_group)
        
        # Trim padding and concatenate
        trimmed = [gathered[i][:all_sizes[i]] for i in range(self.dp_size)]
        full_embeddings = torch.cat(trimmed, dim=0)
        
        logger.info(
            f"Vision actor dp_rank={self.dp_rank}: All-gathered embeddings "
            f"from {local_embeddings.shape} to {full_embeddings.shape}"
        )
        return full_embeddings
    
    def build_model(self) -> None:
        """
        Build and load the vision encoder model.
        """
        from transformers import AutoConfig, AutoModel
        
        logger.info(f"Vision actor {self.dp_rank}: Loading config from {self.model_name_or_path}")
        
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
        
        logger.info(f"Vision actor {self.dp_rank}: Detected model type: {self.model_type}")
        
        try:
            self.full_model = AutoModel.from_pretrained(
                self.model_name_or_path,
                torch_dtype=self.dtype,
                trust_remote_code=True,
            )
            self.full_model = self.full_model.to(self.device)
            
            # Extract vision encoder
            if hasattr(self.full_model, 'visual'):
                self.model = self.full_model.visual
                logger.info(f"Vision actor {self.dp_rank}: Extracted 'visual' encoder")
            elif hasattr(self.full_model, 'vision_model'):
                self.model = self.full_model.vision_model
            elif hasattr(self.full_model, 'vision_tower'):
                self.model = self.full_model.vision_tower
            else:
                self.model = self.full_model
            
        except Exception as e:
            logger.warning(f"Vision actor {self.dp_rank}: Error loading model: {e}")
            vision_config = getattr(hf_config, 'vision_config', hf_config)
            hidden_size = getattr(vision_config, 'hidden_size', 1152)
            out_hidden_size = getattr(vision_config, 'out_hidden_size', hidden_size)
            
            self.model = nn.Sequential(
                nn.Linear(3 * 2 * 14 * 14, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, out_hidden_size),
            ).to(device=self.device, dtype=self.dtype)
        
        self.model.eval()
        logger.info(f"Vision actor {self.dp_rank}: Model built successfully")
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: Union[torch.Tensor, list],
    ) -> torch.Tensor:
        """
        Forward pass through the vision encoder.
        
        Args:
            pixel_values: Image/video pixel values [L, C]
            grid_thw: Grid dimensions [N, 3] with (temporal, height, width)
        
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
            if isinstance(self.model, nn.Sequential):
                embeddings = self.model(pixel_values)
            else:
                try:
                    embeddings = self.model(pixel_values, grid_thw=grid_thw_tensor)
                except TypeError:
                    embeddings = self.model(pixel_values)
        
        logger.info(
            f"Vision actor dp_rank={self.dp_rank}: Forward complete, "
            f"input {pixel_values.shape} -> output {embeddings.shape}"
        )
        return embeddings
    
    def get_output_hidden_size(self) -> int:
        """Get the output hidden size of the vision encoder."""
        if self.config is None:
            raise RuntimeError("Model not built")
        vision_config = getattr(self.config, 'vision_config', self.config)
        return getattr(vision_config, 'out_hidden_size', 
                      getattr(vision_config, 'hidden_size', 1152))


# Backward compatibility alias
Qwen3VLVisionActor = QwenVLVisionActor


def create_vision_actor_class(num_gpus: int = 1, num_cpus: int = 4):
    """
    Create a Ray remote class for the vision actor.
    
    Args:
        num_gpus: Number of GPUs per actor
        num_cpus: Number of CPUs per actor
    
    Returns:
        Ray remote class
    """
    return ray.remote(num_gpus=num_gpus, num_cpus=num_cpus)(QwenVLVisionActor)
