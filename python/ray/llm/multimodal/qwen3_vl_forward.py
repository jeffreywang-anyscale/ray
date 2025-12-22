"""
Qwen3VL Multimodal Forward Pass with Scalable Vision DP + LLM TP

This module orchestrates the multimodal inference:
- Vision encoder: Data Parallelism (DP) with configurable number of actors
- Language model: Tensor Parallelism (TP) with one actor per TP rank

The key insight is that vision encoders can scale independently from the
language model's TP configuration. For example:
- 8 vision DP actors + 4 LLM TP actors
- 16 vision DP actors + 8 LLM TP actors

Usage:
    python qwen3_vl_forward.py --model Qwen/Qwen3-VL-2B \
        --num_vision_actors 4 --tp_size 2
"""

import argparse
import logging
import os
from typing import List, Optional, Tuple

import ray
import torch

from actor_group import ActorGroup
from qwen3_vl_vision_actor import Qwen3VLVisionActor, create_vision_actor_class
from qwen3_vl_llm_actor import Qwen3VLLLMActor, create_llm_actor_class

logger = logging.getLogger(__name__)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class Qwen3VLDistributedModel:
    """
    Distributed Qwen3VL model with scalable vision encoders and TP language model.
    
    This class manages:
    - Multiple vision encoder actors for data parallelism
    - Multiple language model actors for tensor parallelism
    
    Args:
        model_name_or_path: HuggingFace model name or local path
        num_vision_actors: Number of vision encoder actors (DP)
        tp_size: Tensor parallel size for language model
        dtype: Model dtype
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        num_vision_actors: int = 2,
        tp_size: int = 1,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.model_name_or_path = model_name_or_path
        self.num_vision_actors = num_vision_actors
        self.tp_size = tp_size
        self.dtype = dtype
        
        self.vision_actors: List[ray.actor.ActorHandle] = []
        self.llm_actors: List[ray.actor.ActorHandle] = []
        
        # Port for distributed initialization
        self.master_port = self._find_free_port()
    
    @staticmethod
    def _find_free_port() -> int:
        """Find a free port for distributed communication."""
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            return s.getsockname()[1]
    
    def setup(self) -> None:
        """Set up all actors and build models."""
        logger.info(
            f"Setting up {self.num_vision_actors} vision actors (DP) and "
            f"{self.tp_size} LLM actors (TP)"
        )
        
        # Create vision actors
        VisionActorClass = create_vision_actor_class(num_gpus=1, num_cpus=4)
        for i in range(self.num_vision_actors):
            actor = VisionActorClass.remote(
                model_name_or_path=self.model_name_or_path,
                rank=i,
                dtype=self.dtype,
            )
            self.vision_actors.append(actor)
        
        # Create LLM actors
        LLMActorClass = create_llm_actor_class(num_gpus=1, num_cpus=4)
        for i in range(self.tp_size):
            actor = LLMActorClass.remote(
                model_name_or_path=self.model_name_or_path,
                tp_rank=i,
                tp_size=self.tp_size,
                master_addr="127.0.0.1",
                master_port=self.master_port,
                dtype=self.dtype,
            )
            self.llm_actors.append(actor)
        
        # Build all models in parallel
        logger.info("Building vision models...")
        vision_build_refs = [actor.build_model.remote() for actor in self.vision_actors]
        
        logger.info("Building LLM models...")
        llm_build_refs = [actor.build_model.remote() for actor in self.llm_actors]
        
        # Wait for all builds to complete
        ray.get(vision_build_refs + llm_build_refs)
        
        logger.info("All models built successfully!")
        logger.info(f"  Vision actors: {self.num_vision_actors} (each on 1 GPU)")
        logger.info(f"  LLM actors: {self.tp_size} (TP across {self.tp_size} GPUs)")
        logger.info(f"  Total GPUs: {self.num_vision_actors + self.tp_size}")
    
    def split_batch(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Split a batch of images across vision actors for data parallelism.
        
        The input batch is divided into N chunks where N = num_vision_actors.
        Each vision actor will process one chunk.
        
        Args:
            pixel_values: Full batch of pixel values [total_patches, patch_dim]
            grid_thw: Grid dimensions [num_images, 3] with (t, h, w) per image
        
        Returns:
            Tuple of (pixel_values_chunks, grid_thw_chunks) split across actors
        """
        num_images = grid_thw.shape[0]
        
        if num_images < self.num_vision_actors:
            raise ValueError(
                f"Batch has {num_images} images but {self.num_vision_actors} vision actors. "
                f"Need at least as many images as actors."
            )
        
        # Calculate images per actor
        images_per_actor = num_images // self.num_vision_actors
        remainder = num_images % self.num_vision_actors
        
        # Split grid_thw first to know how to split pixel_values
        grid_thw_chunks = []
        pixel_values_chunks = []
        
        image_idx = 0
        patch_idx = 0
        
        for actor_idx in range(self.num_vision_actors):
            # This actor gets images_per_actor images, plus 1 extra if actor_idx < remainder
            num_images_for_actor = images_per_actor + (1 if actor_idx < remainder else 0)
            
            # Get the grid_thw slice for this actor
            actor_grid_thw = grid_thw[image_idx : image_idx + num_images_for_actor]
            grid_thw_chunks.append(actor_grid_thw)
            
            # Calculate number of patches for this actor's images
            # patches = t * h * w for each image
            num_patches = int(actor_grid_thw.prod(dim=1).sum().item())
            
            # Get the pixel_values slice for this actor
            actor_pixel_values = pixel_values[patch_idx : patch_idx + num_patches]
            pixel_values_chunks.append(actor_pixel_values)
            
            image_idx += num_images_for_actor
            patch_idx += num_patches
        
        logger.info(
            f"Split batch of {num_images} images into {self.num_vision_actors} chunks: "
            f"{[g.shape[0] for g in grid_thw_chunks]} images per actor"
        )
        
        return pixel_values_chunks, grid_thw_chunks
    
    def forward_vision(
        self,
        pixel_values_chunks: List[torch.Tensor],
        grid_thw_chunks: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Forward pass through vision encoders with data parallelism.
        
        Each vision actor processes one chunk of the batch.
        
        Args:
            pixel_values_chunks: List of pixel values chunks, one per vision actor
            grid_thw_chunks: List of grid dimensions chunks, one per vision actor
        
        Returns:
            List of vision embeddings from each actor
        """
        if len(pixel_values_chunks) != self.num_vision_actors:
            raise ValueError(
                f"Expected {self.num_vision_actors} chunks, got {len(pixel_values_chunks)}"
            )
        
        # Dispatch to vision actors in parallel
        # Each actor processes a different portion of the batch (Data Parallelism)
        refs = []
        for i, actor in enumerate(self.vision_actors):
            ref = actor.forward.remote(pixel_values_chunks[i], grid_thw_chunks[i])
            refs.append(ref)
        
        # Gather results from all actors
        results = ray.get(refs)
        return results
    
    def forward_llm(
        self,
        vision_embeddings: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through language model with tensor parallelism.
        
        IMPORTANT: All TP ranks receive the SAME full concatenated vision embeddings.
        This is Tensor Parallelism - each rank has a shard of the model weights,
        but they all process the same input data. They coordinate via NCCL
        all-reduce operations during the forward pass.
        
        Args:
            vision_embeddings: Full concatenated vision embeddings from all vision actors
                              Shape: [total_tokens, hidden_size]
            input_ids: Optional text token ids
            attention_mask: Optional attention mask
        
        Returns:
            Language model output from rank 0
        """
        logger.info(
            f"Broadcasting vision embeddings (shape: {vision_embeddings.shape}) "
            f"to all {self.tp_size} TP ranks"
        )
        
        # All TP actors receive the SAME full input (Tensor Parallelism)
        # Each actor holds a shard of the model weights, not a shard of the data
        refs = []
        for tp_rank, actor in enumerate(self.llm_actors):
            # Same vision_embeddings goes to every TP rank
            ref = actor.forward.remote(
                vision_embeddings,  # Full batch, not sharded
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            refs.append(ref)
            logger.debug(f"Dispatched to TP rank {tp_rank}")
        
        # Wait for all actors - they coordinate internally via NCCL
        # Return rank 0's result (all ranks produce the same output after all-reduce)
        results = ray.get(refs)
        return results[0]
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Full multimodal forward pass with automatic batch splitting.
        
        Data flow:
        1. Input batch is SPLIT across vision actors (Data Parallelism)
           - Each vision actor processes a portion of the images
        2. Vision outputs are CONCATENATED
        3. Full concatenated embeddings go to ALL TP ranks (Tensor Parallelism)
           - Each TP rank has weight shards, but processes full data
        
        Args:
            pixel_values: Full batch of pixel values [total_patches, patch_dim]
            grid_thw: Grid dimensions [num_images, 3] with (t, h, w) per image
            input_ids: Optional text token ids
            attention_mask: Optional attention mask
        
        Returns:
            Language model output
        """
        # Step 1: Split batch across vision actors
        logger.info(f"Splitting batch of {grid_thw.shape[0]} images across {self.num_vision_actors} vision actors...")
        pixel_values_chunks, grid_thw_chunks = self.split_batch(pixel_values, grid_thw)
        
        # Step 2: Vision forward (Data Parallelism)
        # Each actor processes its chunk independently
        logger.info("Running vision forward pass (DP)...")
        vision_outputs = self.forward_vision(pixel_values_chunks, grid_thw_chunks)
        
        # Step 3: Concatenate vision outputs from all actors
        combined_embeddings = torch.cat(vision_outputs, dim=0)
        logger.info(f"Concatenated vision embeddings shape: {combined_embeddings.shape}")
        
        # Step 4: LLM forward (Tensor Parallelism)
        # Full concatenated embeddings go to ALL TP ranks
        logger.info("Running LLM forward pass (TP)...")
        output = self.forward_llm(
            combined_embeddings,  # Full batch to all TP ranks
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        return output
    
    def forward_pre_split(
        self,
        pixel_values_chunks: List[torch.Tensor],
        grid_thw_chunks: List[torch.Tensor],
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with pre-split inputs (for advanced usage).
        
        Use this if you've already split the batch yourself.
        
        Args:
            pixel_values_chunks: Pre-split pixel values, one chunk per vision actor
            grid_thw_chunks: Pre-split grid dimensions, one chunk per vision actor
            input_ids: Optional text token ids
            attention_mask: Optional attention mask
        
        Returns:
            Language model output
        """
        # Vision forward (DP)
        vision_outputs = self.forward_vision(pixel_values_chunks, grid_thw_chunks)
        
        # Concatenate and send to LLM (TP)
        combined_embeddings = torch.cat(vision_outputs, dim=0)
        output = self.forward_llm(
            combined_embeddings,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        return output
    
    def shutdown(self) -> None:
        """Shutdown all actors."""
        for actor in self.vision_actors + self.llm_actors:
            ray.kill(actor)
        self.vision_actors = []
        self.llm_actors = []
        logger.info("All actors shutdown")


def create_dummy_batch(
    num_images: int = 8,
    patch_size: int = 14,
    temporal_patch_size: int = 2,
    grid_h: int = 16,
    grid_w: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a dummy batch of images for testing.
    
    This creates a single batch that will be automatically split across
    vision actors by the model.
    
    Args:
        num_images: Total number of images in the batch
        patch_size: Patch size
        temporal_patch_size: Temporal patch size
        grid_h: Grid height (in patches)
        grid_w: Grid width (in patches)
    
    Returns:
        Tuple of (pixel_values, grid_thw) as single tensors
    """
    # Each image has grid_h * grid_w patches
    patches_per_image = grid_h * grid_w
    total_patches = num_images * patches_per_image
    
    # Patch dimension: in_channels * temporal_patch_size * patch_size * patch_size
    patch_dim = 3 * temporal_patch_size * patch_size * patch_size
    
    # Create pixel values: [total_patches, patch_dim]
    pixel_values = torch.randn(total_patches, patch_dim)
    
    # Create grid_thw: [num_images, 3] with (t, h, w) per image
    # All images have same dimensions: 1 frame, grid_h x grid_w
    grid_thw = torch.tensor([[1, grid_h, grid_w]] * num_images)
    
    return pixel_values, grid_thw


def main():
    parser = argparse.ArgumentParser(
        description="Run Qwen3VL with scalable vision DP + LLM TP"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--num_vision_actors",
        type=int,
        default=2,
        help="Number of vision encoder actors for data parallelism",
    )
    parser.add_argument(
        "--tp_size",
        type=int,
        default=1,
        help="Tensor parallel size for language model",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=8,
        help="Number of images in the batch",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype",
    )
    args = parser.parse_args()
    
    # Parse dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init()
    
    print("=" * 60)
    print("Qwen3VL Multimodal Forward Pass Demo")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Vision actors (DP): {args.num_vision_actors}")
    print(f"LLM actors (TP): {args.tp_size}")
    print(f"Total GPUs: {args.num_vision_actors + args.tp_size}")
    print(f"Batch size: {args.num_images} images")
    print("=" * 60)
    print()
    print("Data Flow:")
    print(f"  1. Input: {args.num_images} images")
    print(f"  2. SPLIT across {args.num_vision_actors} vision actors (DP)")
    print(f"     -> Each actor processes ~{args.num_images // args.num_vision_actors} images")
    print(f"  3. CONCATENATE vision outputs")
    print(f"  4. BROADCAST full embeddings to all {args.tp_size} TP ranks")
    print(f"     -> Each TP rank receives the SAME full batch")
    print("=" * 60)
    
    # Create distributed model
    model = Qwen3VLDistributedModel(
        model_name_or_path=args.model,
        num_vision_actors=args.num_vision_actors,
        tp_size=args.tp_size,
        dtype=dtype,
    )
    
    # Setup actors
    model.setup()
    
    # Create dummy batch (single batch that will be auto-split)
    print("\nCreating dummy batch...")
    pixel_values, grid_thw = create_dummy_batch(num_images=args.num_images)
    print(f"  Total images: {grid_thw.shape[0]}")
    print(f"  Pixel values shape: {pixel_values.shape}")
    print(f"  Grid THW shape: {grid_thw.shape}")
    
    # Run forward pass - the model will automatically split the batch
    print("\nRunning forward pass...")
    output = model.forward(
        pixel_values=pixel_values,
        grid_thw=grid_thw,
    )
    
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    print(f"Output shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")
    print(f"Output device: {output.device}")
    
    # Cleanup
    model.shutdown()
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()

