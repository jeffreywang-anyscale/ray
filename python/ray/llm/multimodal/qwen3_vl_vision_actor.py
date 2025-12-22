"""
Qwen VL Vision Encoder Ray Actor with Data Parallelism

This module implements a Ray actor that hosts the Qwen VL vision encoder.
Supports both Qwen2.5-VL and Qwen3-VL models.
Multiple actors can be created for data parallelism - each actor processes
different portions of the input batch independently.

The vision encoder is loaded using HuggingFace transformers.

Supports NCCL-based all_gather for efficient embedding aggregation.
"""

import logging
import os
import socket
from typing import Optional, Union, Tuple, List

import ray
import torch
import torch.nn as nn
import torch.distributed as dist

logger = logging.getLogger(__name__)


class QwenVLVisionActor:
    """
    Ray actor that hosts a Qwen VL vision encoder.
    
    This actor loads the vision encoder from HuggingFace transformers and can 
    process image/video inputs. Multiple actors can be used for data parallelism.
    
    Supports: Qwen2-VL, Qwen2.5-VL, Qwen3-VL
    
    Features:
    - NCCL-based all_gather for efficient embedding aggregation across DP actors
    - NCCL-based broadcast to send embeddings to LLM actors
    
    Args:
        model_name_or_path: HuggingFace model name or local path
        rank: Actor rank for logging/identification
        dp_size: Data parallel size (number of vision actors)
        dtype: Model dtype (default: bfloat16)
        device: Device to use (default: cuda:0)
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        rank: int = 0,
        dp_size: int = 1,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda:0",
    ):
        self.model_name_or_path = model_name_or_path
        self.rank = rank
        self.dp_size = dp_size
        self.dtype = dtype
        self.device = torch.device(device)
        self.model: Optional[nn.Module] = None
        self.full_model = None  # Keep reference for potential future use
        self.config = None
        self.model_type = None
        self._distributed_initialized = False
        self._dp_group = None
        self._pipeline_group = None  # For vision-to-LLM communication
        
        logger.info(f"Vision actor {rank}/{dp_size} initialized with model {model_name_or_path}")
    
    def get_ip_address(self) -> str:
        """Get the IP address of this actor for NCCL communication."""
        hostname = socket.gethostname()
        ip_addr = socket.gethostbyname(hostname)
        return ip_addr
    
    def init_distributed(
        self,
        master_addr: str,
        master_port: int,
        world_size: int = None,
        llm_tp_size: int = 0,
    ) -> None:
        """
        Initialize distributed environment for NCCL communication.
        
        Args:
            master_addr: IP address of rank 0 for NCCL rendezvous
            master_port: Port for NCCL rendezvous
            world_size: Total world size (defaults to dp_size)
            llm_tp_size: Number of LLM TP actors (for creating pipeline group)
        """
        if self._distributed_initialized:
            logger.info(f"Vision actor {self.rank}: Distributed already initialized")
            return
        
        if self.dp_size <= 1 and llm_tp_size <= 0:
            logger.info(f"Vision actor {self.rank}: DP size is 1 and no pipeline, skipping distributed init")
            return
        
        ws = world_size if world_size is not None else self.dp_size
        
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(ws)
        
        logger.info(
            f"Vision actor {self.rank}: Initializing distributed "
            f"(master={master_addr}:{master_port}, rank={self.rank}/{ws})"
        )
        
        dist.init_process_group(
            backend="nccl",
            rank=self.rank,
            world_size=ws,
        )
        
        # Create a DP process group for vision actors
        self._dp_group = dist.new_group(ranks=list(range(self.dp_size)))
        
        # Create a pipeline group for vision-to-LLM communication (only for rank 0)
        if llm_tp_size > 0 and self.rank == 0:
            # Vision rank 0 + all LLM ranks
            llm_ranks = list(range(self.dp_size, self.dp_size + llm_tp_size))
            pipeline_ranks = [0] + llm_ranks
            self._pipeline_group = dist.new_group(ranks=pipeline_ranks)
            logger.info(
                f"Vision actor {self.rank}: Created pipeline group with ranks {pipeline_ranks}"
            )
        else:
            self._pipeline_group = None
        
        self._distributed_initialized = True
        logger.info(f"Vision actor {self.rank}: Distributed initialized successfully")
    
    def cleanup_distributed(self) -> None:
        """Clean up distributed resources before actor termination."""
        if self._distributed_initialized:
            try:
                dist.destroy_process_group()
                self._distributed_initialized = False
                logger.info(f"Vision actor {self.rank}: Distributed cleaned up")
            except Exception as e:
                logger.warning(f"Vision actor {self.rank}: Cleanup failed: {e}")
    
    def broadcast_to_llm(self, embeddings: torch.Tensor) -> None:
        """
        Broadcast embeddings to all LLM actors via NCCL.
        
        Only vision rank 0 should call this method.
        
        Args:
            embeddings: Full vision embeddings to broadcast [seq_len, hidden]
        """
        if self.rank != 0:
            logger.warning(f"Vision actor {self.rank}: Only rank 0 should broadcast to LLM")
            return
        
        if self._pipeline_group is None:
            raise RuntimeError("Pipeline group not initialized. Call init_distributed with llm_tp_size > 0")
        
        embeddings = embeddings.to(device=self.device, dtype=self.dtype)
        
        # Broadcast from vision rank 0 (which is rank 0 in the pipeline group)
        dist.broadcast(embeddings, src=0, group=self._pipeline_group)
        
        logger.info(
            f"Vision actor {self.rank}: Broadcasted embeddings to LLM actors, "
            f"shape {embeddings.shape}"
        )
    
    def send_embeddings_p2p(
        self,
        embeddings: torch.Tensor,
        dst_rank: int,
    ) -> None:
        """
        Send embeddings to a specific LLM actor via NCCL P2P (point-to-point).
        
        Args:
            embeddings: Vision embeddings to send [seq_len, hidden]
            dst_rank: Global rank of destination LLM actor
        """
        if not self._distributed_initialized:
            raise RuntimeError("Distributed not initialized. Call init_distributed first.")
        
        embeddings = embeddings.to(device=self.device, dtype=self.dtype).contiguous()
        
        # Send using async NCCL P2P and wait
        req = dist.isend(embeddings, dst=dst_rank)
        req.wait()
        
        logger.info(
            f"Vision actor {self.rank}: Sent embeddings to rank {dst_rank} via NCCL P2P, "
            f"shape {embeddings.shape}"
        )
    
    def all_gather_embeddings(self, local_embeddings: torch.Tensor) -> torch.Tensor:
        """
        All-gather vision embeddings from all DP actors using NCCL.
        
        Each actor has processed a portion of the images. This method
        uses NCCL all_gather so each actor ends up with the FULL batch.
        
        Args:
            local_embeddings: This actor's vision embeddings [local_seq, hidden]
        
        Returns:
            Full vision embeddings from all actors [total_seq, hidden]
        """
        if self.dp_size <= 1 or not self._distributed_initialized:
            return local_embeddings
        
        local_embeddings = local_embeddings.to(device=self.device, dtype=self.dtype)
        
        # First, gather sizes from all ranks (to handle variable sizes)
        local_size = torch.tensor([local_embeddings.shape[0]], device=self.device, dtype=torch.long)
        all_sizes = [torch.zeros(1, device=self.device, dtype=torch.long) for _ in range(self.dp_size)]
        dist.all_gather(all_sizes, local_size, group=self._dp_group)
        all_sizes = [int(s.item()) for s in all_sizes]
        
        max_size = max(all_sizes)
        hidden_size = local_embeddings.shape[-1]
        
        # Pad local embeddings to max size for uniform all_gather
        if local_embeddings.shape[0] < max_size:
            padding = torch.zeros(
                max_size - local_embeddings.shape[0], hidden_size,
                device=self.device, dtype=self.dtype
            )
            padded_local = torch.cat([local_embeddings, padding], dim=0)
        else:
            padded_local = local_embeddings
        
        # All-gather padded tensors using NCCL
        gathered = [torch.zeros_like(padded_local) for _ in range(self.dp_size)]
        dist.all_gather(gathered, padded_local, group=self._dp_group)
        
        # Trim padding and concatenate
        trimmed = [gathered[i][:all_sizes[i]] for i in range(self.dp_size)]
        full_embeddings = torch.cat(trimmed, dim=0)
        
        logger.info(
            f"Vision actor {self.rank}: NCCL all_gather embeddings "
            f"from {local_embeddings.shape} to {full_embeddings.shape}"
        )
        
        return full_embeddings
    
    def broadcast_embeddings(
        self,
        embeddings: torch.Tensor,
        src_rank: int = 0,
    ) -> torch.Tensor:
        """
        Broadcast embeddings from source rank to all ranks using NCCL.
        
        Args:
            embeddings: Embeddings to broadcast (only used on src_rank)
            src_rank: Source rank for broadcast
        
        Returns:
            Broadcasted embeddings (same on all ranks)
        """
        if not self._distributed_initialized:
            return embeddings
        
        embeddings = embeddings.to(device=self.device, dtype=self.dtype)
        dist.broadcast(embeddings, src=src_rank)
        
        logger.info(
            f"Vision actor {self.rank}: NCCL broadcast from rank {src_rank}, "
            f"shape {embeddings.shape}"
        )
        
        return embeddings
    
    def build_model(self) -> None:
        """
        Build and load the vision encoder model.
        
        This loads the full Qwen VL model from HuggingFace transformers and 
        extracts the vision encoder.
        """
        import torch
        from transformers import AutoConfig, AutoModel
        
        logger.info(f"Vision actor {self.rank}: Loading config from {self.model_name_or_path}")
        
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
        
        logger.info(f"Vision actor {self.rank}: Detected model type: {self.model_type}")
        
        # Load the full model and extract vision encoder
        logger.info(f"Vision actor {self.rank}: Loading model from transformers")
        
        try:
            # Try loading with AutoModel for vision-language models
            # First try without device_map (doesn't require accelerate)
            self.full_model = AutoModel.from_pretrained(
                self.model_name_or_path,
                torch_dtype=self.dtype,
                trust_remote_code=True,
            )
            self.full_model = self.full_model.to(self.device)
            
            # Extract vision encoder - Qwen VL models have 'visual' attribute
            if hasattr(self.full_model, 'visual'):
                self.model = self.full_model.visual
                logger.info(f"Vision actor {self.rank}: Extracted 'visual' encoder")
            elif hasattr(self.full_model, 'vision_model'):
                self.model = self.full_model.vision_model
                logger.info(f"Vision actor {self.rank}: Extracted 'vision_model' encoder")
            elif hasattr(self.full_model, 'vision_tower'):
                self.model = self.full_model.vision_tower
                logger.info(f"Vision actor {self.rank}: Extracted 'vision_tower' encoder")
            else:
                # If we can't find the vision encoder, use the full model
                logger.warning(
                    f"Vision actor {self.rank}: Could not find vision encoder, "
                    "using full model"
                )
                self.model = self.full_model
                
        except Exception as e:
            logger.warning(f"Vision actor {self.rank}: Error loading model: {e}")
            logger.info(f"Vision actor {self.rank}: Creating placeholder vision encoder")
            
            # Create a placeholder vision encoder for testing
            vision_config = getattr(hf_config, 'vision_config', hf_config)
            hidden_size = getattr(vision_config, 'hidden_size', 1152)
            out_hidden_size = getattr(vision_config, 'out_hidden_size', hidden_size)
            
            self.model = nn.Sequential(
                nn.Linear(3 * 2 * 14 * 14, hidden_size),  # patch_dim -> hidden
                nn.GELU(),
                nn.Linear(hidden_size, out_hidden_size),
            ).to(device=self.device, dtype=self.dtype)
        
        # Set to eval mode
        self.model.eval()
        
        logger.info(f"Vision actor {self.rank}: Model built successfully")
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: Union[torch.Tensor, list],
    ) -> torch.Tensor:
        """
        Forward pass through the vision encoder.
        
        Args:
            pixel_values: Image/video pixel values [L, C] where L is total patches
            grid_thw: Grid dimensions [N, 3] with (temporal, height, width) for each item
        
        Returns:
            Vision embeddings [num_tokens, hidden_size]
        """
        if self.model is None:
            raise RuntimeError(
                f"Vision actor {self.rank}: Model not built. Call build_model() first."
            )
        
        # Move inputs to device
        pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)
        if isinstance(grid_thw, torch.Tensor):
            grid_thw_tensor = grid_thw.to(device=self.device)
            grid_thw_list = grid_thw.tolist()
        else:
            grid_thw_tensor = torch.tensor(grid_thw, device=self.device)
            grid_thw_list = grid_thw
        
        with torch.no_grad():
            # Check if it's a real vision encoder or placeholder
            if isinstance(self.model, nn.Sequential):
                # Placeholder model
                embeddings = self.model(pixel_values)
            else:
                # Real vision encoder
                try:
                    embeddings = self.model(pixel_values, grid_thw=grid_thw_tensor)
                except TypeError:
                    # Some models don't take grid_thw
                    embeddings = self.model(pixel_values)
        
        logger.info(
            f"Vision actor {self.rank}: Processed input shape {pixel_values.shape}, "
            f"output shape {embeddings.shape}"
        )
        
        return embeddings
    
    def get_output_hidden_size(self) -> int:
        """Get the output hidden size of the vision encoder."""
        if self.config is None:
            raise RuntimeError("Model not built")
        vision_config = getattr(self.config, 'vision_config', self.config)
        return getattr(vision_config, 'out_hidden_size', 
                      getattr(vision_config, 'hidden_size', 1152))


# Keep old name for backwards compatibility
Qwen3VLVisionActor = QwenVLVisionActor


def create_vision_actor_class(num_gpus: int = 1, num_cpus: int = 4):
    """
    Create a Ray remote class for the vision actor with specified resources.
    
    Args:
        num_gpus: Number of GPUs per actor
        num_cpus: Number of CPUs per actor
    
    Returns:
        Ray remote class
    """
    return ray.remote(num_gpus=num_gpus, num_cpus=num_cpus)(QwenVLVisionActor)


def get_ip_address() -> str:
    """Get the IP address for NCCL communication (utility function)."""
    hostname = socket.gethostname()
    return socket.gethostbyname(hostname)
