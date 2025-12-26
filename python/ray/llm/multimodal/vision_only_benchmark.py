#!/usr/bin/env python3
"""
Vision encoder only benchmark.
Measures the maximum req/s a single vision encoder replica can yield.
"""

import argparse
import logging
import time
from typing import Tuple

import ray
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_dummy_batch(
    num_images: int,
    image_height: int = 448,
    image_width: int = 448,
    patch_size: int = 14,
    temporal_patch_size: int = 2,
    in_channels: int = 3,
    spatial_merge_size: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Create a dummy batch of images for benchmarking."""
    h_patches = image_height // patch_size
    w_patches = image_width // patch_size
    t_patches = 1
    
    patches_per_image = t_patches * h_patches * w_patches
    total_patches = num_images * patches_per_image
    visual_tokens_per_image = (t_patches * h_patches * w_patches) // (spatial_merge_size ** 2)
    patch_dim = temporal_patch_size * in_channels * patch_size * patch_size
    
    torch.manual_seed(42)
    pixel_values = torch.randn(total_patches, patch_dim, dtype=torch.float32)
    grid_thw = torch.tensor([[t_patches, h_patches, w_patches]] * num_images, dtype=torch.long)
    
    return pixel_values, grid_thw, visual_tokens_per_image


def benchmark_vision_only(
    model_name: str,
    images_per_batch: int,
    num_batches: int,
    warmup_batches: int,
    image_size: int,
):
    """Benchmark vision encoder only - no LLM, no distributed comm."""
    
    from qwen3_vl_vision_actor import QwenVLVisionActor
    
    # Create single vision actor (DP=1)
    VisionActorCls = ray.remote(num_gpus=1.0)(QwenVLVisionActor)
    vision_actor = VisionActorCls.remote(
        model_name_or_path=model_name,
        rank=0,
        dp_size=1,
    )
    
    # Build model
    logger.info("Building vision encoder...")
    ray.get(vision_actor.build_model.remote())
    logger.info("Vision encoder ready")
    
    # Create batch
    pixel_values, grid_thw, visual_tokens_per_image = create_dummy_batch(
        images_per_batch, 
        image_height=image_size,
        image_width=image_size,
    )
    tokens_per_batch = images_per_batch * visual_tokens_per_image
    
    logger.info(f"Batch: {images_per_batch} images @ {image_size}x{image_size}")
    logger.info(f"Visual tokens per image: {visual_tokens_per_image}")
    logger.info(f"Tokens per batch: {tokens_per_batch}")
    
    # Warmup
    logger.info(f"Warming up ({warmup_batches} batches)...")
    for _ in range(warmup_batches):
        ray.get(vision_actor.forward.remote(pixel_values, grid_thw))
    
    # Benchmark
    logger.info(f"Benchmarking ({num_batches} batches)...")
    times = []
    for i in range(num_batches):
        start = time.perf_counter()
        ray.get(vision_actor.forward.remote(pixel_values, grid_thw))
        end = time.perf_counter()
        times.append(end - start)
        
        if (i + 1) % 50 == 0:
            logger.info(f"  Completed {i + 1}/{num_batches} batches")
    
    # Cleanup
    ray.kill(vision_actor)
    
    # Calculate stats
    total_time = sum(times)
    avg_time = total_time / num_batches
    min_time = min(times)
    max_time = max(times)
    
    req_per_sec = num_batches / total_time
    images_per_sec = (num_batches * images_per_batch) / total_time
    tokens_per_sec = (num_batches * tokens_per_batch) / total_time
    
    # Print results
    print("\n" + "=" * 60)
    print("VISION ENCODER BENCHMARK (Single Replica)")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Image size:           {image_size}x{image_size}")
    print(f"  Images per batch:     {images_per_batch}")
    print(f"  Visual tokens/image:  {visual_tokens_per_image}")
    print(f"  Tokens per batch:     {tokens_per_batch}")
    print(f"  Num batches:          {num_batches}")
    
    print(f"\nLatency (seconds/batch):")
    print(f"  Mean:    {avg_time:.4f}")
    print(f"  Min:     {min_time:.4f}")
    print(f"  Max:     {max_time:.4f}")
    
    print(f"\nThroughput:")
    print(f"  Requests/sec:   {req_per_sec:.2f}")
    print(f"  Images/sec:     {images_per_sec:.2f}")
    print(f"  Tokens/sec:     {tokens_per_sec:.2f}")
    print("=" * 60)
    
    return {
        'req_per_sec': req_per_sec,
        'images_per_sec': images_per_sec,
        'tokens_per_sec': tokens_per_sec,
        'latency_mean': avg_time,
        'latency_min': min_time,
        'latency_max': max_time,
    }


def main():
    parser = argparse.ArgumentParser(description="Vision encoder only benchmark")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--images_per_batch", type=int, default=1, help="Images per request/batch")
    parser.add_argument("--num_batches", type=int, default=100, help="Number of batches to run")
    parser.add_argument("--warmup_batches", type=int, default=10, help="Warmup batches")
    parser.add_argument("--image_size", type=int, default=448, help="Image size (must be divisible by 28)")
    args = parser.parse_args()
    
    # Validate image size
    if args.image_size % 28 != 0:
        print(f"Error: image_size must be divisible by 28. Got {args.image_size}")
        print(f"Valid sizes: 28, 56, 112, 224, 448, 672, 896, 1120, 1344, ...")
        return
    
    if not ray.is_initialized():
        ray.init()
    
    try:
        benchmark_vision_only(
            model_name=args.model_name,
            images_per_batch=args.images_per_batch,
            num_batches=args.num_batches,
            warmup_batches=args.warmup_batches,
            image_size=args.image_size,
        )
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()

