#!/usr/bin/env python3
"""
Benchmark script for Vision DP + LLM TP configurations.

Supports three modes:

1. Non-colocated (default): Vision DP and LLM TP on SEPARATE GPUs
   - DP=2, TP=2 uses 4 GPUs (2 for vision, 2 for LLM)
   - DP=6, TP=2 uses 8 GPUs (6 for vision, 2 for LLM)

2. Colocated (--colocated): Vision DP and LLM TP on SAME GPUs  
   - DP=2, TP=2 uses 2 GPUs (vision[0]+LLM[0] on GPU0, vision[1]+LLM[1] on GPU1)
   - Requires DP == TP

3. Combined (--combined): Full VLM model in single actor
   - Single forward pass through vision + LLM
   - Use --num_replicas for data parallel replicas
   - Good for baseline comparison

Communication (non-combined modes):
- NCCL all_gather for vision embeddings (among vision actors)
- NCCL broadcast or P2P for vision-to-LLM transfer

Example usage:
python benchmark.py --num_batches 10 --images_per_batch 16 --warmup_batches 2 --vision_dp 2 --llm_tp 1
python benchmark.py --num_batches 10 --images_per_batch 16 --warmup_batches 2 --vision_dp 2 --llm_tp 1 --colocated
"""

import argparse
import logging
import time
from typing import Tuple, Dict, Any

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
    """
    Create a dummy batch of images for benchmarking.
    
    Returns:
        pixel_values: [total_patches, patch_dim]
        grid_thw: [num_images, 3]
        visual_tokens_per_image: number of visual tokens output per image
    """
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


def benchmark_non_colocated(
    model_name: str,
    vision_dp: int,
    llm_tp: int,
    num_batches: int,
    images_per_batch: int,
    warmup_batches: int = 10,
    master_port: int = 29500,
) -> Dict[str, Any]:
    """
    Benchmark with vision and LLM on SEPARATE GPUs.
    
    GPU allocation:
    - Vision GPUs: 0, 1, ..., (vision_dp - 1)
    - LLM GPUs: vision_dp, vision_dp + 1, ..., (vision_dp + llm_tp - 1)
    
    Total GPUs: vision_dp + llm_tp
    """
    from qwen3_vl_vision_actor import create_vision_actor_class
    from qwen3_vl_llm_actor import create_llm_actor_class
    
    total_gpus = vision_dp + llm_tp
    world_size = total_gpus
    
    logger.info(f"\n{'='*60}")
    logger.info(f"NON-COLOCATED: Vision DP={vision_dp}, LLM TP={llm_tp}")
    logger.info(f"  Vision GPUs: 0 to {vision_dp - 1}")
    logger.info(f"  LLM GPUs: {vision_dp} to {total_gpus - 1}")
    logger.info(f"  Total GPUs: {total_gpus}")
    logger.info(f"  Batches: {num_batches} (warmup: {warmup_batches})")
    logger.info(f"  Images per batch: {images_per_batch}")
    logger.info(f"{'='*60}")
    
    # Create vision actors
    VisionActorClass = create_vision_actor_class(num_gpus=1, num_cpus=4)
    vision_actors = [
        VisionActorClass.remote(
            model_name,
            dp_rank=i,
            dp_size=vision_dp,
            global_rank=i,
        )
        for i in range(vision_dp)
    ]
    
    # Create LLM actors
    LLMActorClass = create_llm_actor_class(num_gpus=1, num_cpus=4)
    llm_actors = [
        LLMActorClass.remote(
            model_name,
            tp_rank=i,
            tp_size=llm_tp,
            global_rank=vision_dp + i,
        )
        for i in range(llm_tp)
    ]
    
    # Get master address from vision rank 0
    master_addr = ray.get(vision_actors[0].get_ip_address.remote())
    
    # Initialize distributed for ALL actors
    vision_init_refs = [
        actor.init_distributed.remote(
            master_addr, master_port,
            world_size=world_size,
            llm_tp_size=llm_tp,
        )
        for actor in vision_actors
    ]
    
    llm_init_refs = [
        actor.init_distributed.remote(
            master_addr, master_port,
            world_size=world_size,
            vision_dp_size=vision_dp,
        )
        for actor in llm_actors
    ]
    
    ray.get(vision_init_refs + llm_init_refs)
    logger.info(f"Distributed initialized (world_size={world_size})")
    
    # Build models
    ray.get([actor.build_model.remote() for actor in vision_actors])
    ray.get([actor.build_model.remote() for actor in llm_actors])
    logger.info(f"All models built")
    
    # Create batch
    pixel_values, grid_thw, visual_tokens_per_image = create_dummy_batch(images_per_batch)
    
    # Pre-compute chunk boundaries for DP split
    num_images = grid_thw.shape[0]
    images_per_actor = num_images // vision_dp
    extra = num_images % vision_dp
    patches_per_image = grid_thw[0, 0].item() * grid_thw[0, 1].item() * grid_thw[0, 2].item()
    
    chunks = []
    pixel_offset = 0
    image_offset = 0
    for i in range(vision_dp):
        num_imgs = images_per_actor + (1 if i < extra else 0)
        num_patches = num_imgs * patches_per_image
        chunks.append({
            'pixel_start': pixel_offset,
            'pixel_end': pixel_offset + num_patches,
            'image_start': image_offset,
            'image_end': image_offset + num_imgs,
        })
        pixel_offset += num_patches
        image_offset += num_imgs
    
    # Timing
    vision_times = []
    all_gather_times = []
    transfer_times = []
    llm_times = []
    total_times = []
    
    total_batches = warmup_batches + num_batches
    logger.info(f"Starting benchmark ({warmup_batches} warmup + {num_batches} measured)...")
    
    for batch_idx in range(total_batches):
        is_warmup = batch_idx < warmup_batches
        batch_start = time.perf_counter()
        
        # === VISION FORWARD (DP) ===
        vision_start = time.perf_counter()
        vision_refs = []
        for i in range(vision_dp):
            chunk = chunks[i]
            chunk_pixels = pixel_values[chunk['pixel_start']:chunk['pixel_end']]
            chunk_grid = grid_thw[chunk['image_start']:chunk['image_end']]
            vision_refs.append(vision_actors[i].forward.remote(chunk_pixels, chunk_grid))
        
        local_vision_outputs = ray.get(vision_refs)
        vision_end = time.perf_counter()
        
        # === ALL GATHER (among vision actors) ===
        all_gather_start = time.perf_counter()
        if vision_dp > 1:
            all_gather_refs = [
                vision_actors[i].all_gather_embeddings.remote(local_vision_outputs[i])
                for i in range(vision_dp)
            ]
            full_vision_outputs = ray.get(all_gather_refs)
            vision_embeddings = full_vision_outputs[0]
        else:
            vision_embeddings = local_vision_outputs[0]
        all_gather_end = time.perf_counter()
        
        # === TRANSFER (P2P from vision[0] to all LLM actors) ===
        transfer_start = time.perf_counter()
        embedding_shape = (vision_embeddings.shape[0], vision_embeddings.shape[1])
        
        send_refs = []
        recv_refs = []
        for i in range(llm_tp):
            dst_rank = vision_dp + i
            send_refs.append(vision_actors[0].send_embeddings_p2p.remote(vision_embeddings, dst_rank))
            recv_refs.append(llm_actors[i].recv_embeddings_p2p.remote(embedding_shape, src_rank=0))
        
        ray.get(send_refs)
        received_embeddings = ray.get(recv_refs)
        transfer_end = time.perf_counter()
        
        # === LLM FORWARD (TP) ===
        llm_start = time.perf_counter()
        llm_refs = [
            actor.forward.remote(received_embeddings[i])
            for i, actor in enumerate(llm_actors)
        ]
        ray.get(llm_refs)
        llm_end = time.perf_counter()
        
        batch_end = time.perf_counter()
        
        if not is_warmup:
            vision_times.append(vision_end - vision_start)
            all_gather_times.append(all_gather_end - all_gather_start)
            transfer_times.append(transfer_end - transfer_start)
            llm_times.append(llm_end - llm_start)
            total_times.append(batch_end - batch_start)
        
        if (batch_idx + 1) % 100 == 0:
            status = "warmup" if is_warmup else "measured"
            logger.info(f"  Batch {batch_idx + 1}/{total_batches} ({status})")
    
    # Cleanup
    cleanup_refs = []
    for actor in vision_actors:
        cleanup_refs.append(actor.cleanup_distributed.remote())
    for actor in llm_actors:
        cleanup_refs.append(actor.cleanup_distributed.remote())
    ray.get(cleanup_refs)
    
    for actor in vision_actors + llm_actors:
        ray.kill(actor)
    time.sleep(1)
    
    # Compute statistics
    def stats(times):
        t = torch.tensor(times)
        return {
            'mean': t.mean().item(),
            'std': t.std().item(),
            'min': t.min().item(),
            'max': t.max().item(),
            'p50': t.median().item(),
            'p99': t.quantile(0.99).item(),
        }
    
    results = {
        'mode': 'non-colocated',
        'vision_dp': vision_dp,
        'llm_tp': llm_tp,
        'total_gpus': total_gpus,
        'num_batches': num_batches,
        'images_per_batch': images_per_batch,
        'vision': stats(vision_times),
        'all_gather': stats(all_gather_times),
        'transfer': stats(transfer_times),
        'llm': stats(llm_times),
        'total': stats(total_times),
    }
    
    # Throughput calculations
    total_images = num_batches * images_per_batch
    total_time = sum(total_times)
    total_vision_time = sum(vision_times)
    total_llm_time = sum(llm_times)
    num_visual_tokens_per_batch = visual_tokens_per_image * images_per_batch
    total_visual_tokens = num_batches * num_visual_tokens_per_batch
    
    results['throughput'] = {
        # Legacy compatibility
        'images_per_sec': total_images / total_time,
        'batches_per_sec': num_batches / total_time,
        
        # Vision encoder metrics
        'vision_requests_per_sec': num_batches / total_vision_time,
        'vision_images_per_sec': total_images / total_vision_time,
        'vision_tokens_per_sec': total_visual_tokens / total_vision_time,
        
        # LLM decoder metrics
        'llm_requests_per_sec': num_batches / total_llm_time,
        'llm_images_per_sec': total_images / total_llm_time,
        'llm_tokens_per_sec': total_visual_tokens / total_llm_time,
        
        # End-to-end metrics
        'e2e_requests_per_sec': num_batches / total_time,
        'e2e_images_per_sec': total_images / total_time,
        'e2e_tokens_per_sec': total_visual_tokens / total_time,
    }
    
    results['tokens'] = {
        'visual_tokens_per_image': visual_tokens_per_image,
        'visual_tokens_per_batch': num_visual_tokens_per_batch,
    }
    
    return results


def benchmark_colocated(
    model_name: str,
    dp_tp_size: int,
    num_batches: int,
    images_per_batch: int,
    warmup_batches: int = 10,
    master_port: int = 29500,
) -> Dict[str, Any]:
    """
    Benchmark with vision and LLM on SAME GPUs.
    
    In colocated mode, DP == TP.
    Each GPU has: vision encoder replica + LLM TP shard
    
    GPU allocation:
    - GPU 0: Vision[0] + LLM[0]
    - GPU 1: Vision[1] + LLM[1]
    - ...
    
    Total GPUs: dp_tp_size
    
    NOTE: In colocated mode, LLM actors don't use NCCL - they receive
    embeddings directly from vision actors via Ray and run independently.
    Vision actors handle all NCCL communication (all_gather for DP).
    """
    from qwen3_vl_vision_actor import create_vision_actor_class
    from qwen3_vl_llm_actor import create_llm_actor_class
    
    vision_dp = dp_tp_size
    llm_tp = dp_tp_size
    total_gpus = dp_tp_size
    
    logger.info(f"\n{'='*60}")
    logger.info(f"COLOCATED: Vision DP={vision_dp}, LLM TP={llm_tp}")
    logger.info(f"  GPU 0: Vision[0] + LLM[0]")
    if dp_tp_size > 1:
        logger.info(f"  GPU 1: Vision[1] + LLM[1]")
    if dp_tp_size > 2:
        logger.info(f"  ... (total {dp_tp_size} GPUs)")
    logger.info(f"  Total GPUs: {total_gpus}")
    logger.info(f"  Batches: {num_batches} (warmup: {warmup_batches})")
    logger.info(f"  Images per batch: {images_per_batch}")
    logger.info(f"{'='*60}")
    
    # Create vision actors (one per GPU)
    VisionActorClass = create_vision_actor_class(num_gpus=1, num_cpus=4)
    vision_actors = [
        VisionActorClass.remote(
            model_name,
            dp_rank=i,
            dp_size=vision_dp,
            global_rank=i,
        )
        for i in range(vision_dp)
    ]
    
    # Create LLM actors (share GPU with vision via low resource request)
    # They receive embeddings via Ray, not NCCL
    LLMActorClass = create_llm_actor_class(num_gpus=0.5, num_cpus=2)  # Share GPU
    llm_actors = [
        LLMActorClass.remote(
            model_name,
            tp_rank=i,
            tp_size=1,  # Each LLM runs independently in colocated mode
            global_rank=i,
        )
        for i in range(llm_tp)
    ]
    
    # Only vision actors need distributed (for DP all_gather)
    if vision_dp > 1:
        master_addr = ray.get(vision_actors[0].get_ip_address.remote())
        init_refs = [
            actor.init_distributed.remote(
                master_addr, master_port,
                world_size=vision_dp,
                llm_tp_size=0,
            )
            for actor in vision_actors
        ]
        ray.get(init_refs)
        logger.info(f"Vision distributed initialized (world_size={vision_dp})")
    
    # Build models
    ray.get([actor.build_model.remote() for actor in vision_actors])
    ray.get([actor.build_model.remote() for actor in llm_actors])
    logger.info(f"All models built")
    
    # Create batch
    pixel_values, grid_thw, visual_tokens_per_image = create_dummy_batch(images_per_batch)
    
    # Pre-compute chunk boundaries
    num_images = grid_thw.shape[0]
    images_per_actor = num_images // vision_dp
    extra = num_images % vision_dp
    patches_per_image = grid_thw[0, 0].item() * grid_thw[0, 1].item() * grid_thw[0, 2].item()
    
    chunks = []
    pixel_offset = 0
    image_offset = 0
    for i in range(vision_dp):
        num_imgs = images_per_actor + (1 if i < extra else 0)
        num_patches = num_imgs * patches_per_image
        chunks.append({
            'pixel_start': pixel_offset,
            'pixel_end': pixel_offset + num_patches,
            'image_start': image_offset,
            'image_end': image_offset + num_imgs,
        })
        pixel_offset += num_patches
        image_offset += num_imgs
    
    # Timing
    vision_times = []
    all_gather_times = []
    llm_times = []
    total_times = []
    
    total_batches = warmup_batches + num_batches
    logger.info(f"Starting benchmark ({warmup_batches} warmup + {num_batches} measured)...")
    
    for batch_idx in range(total_batches):
        is_warmup = batch_idx < warmup_batches
        batch_start = time.perf_counter()
        
        # === VISION FORWARD (DP) ===
        vision_start = time.perf_counter()
        vision_refs = []
        for i in range(vision_dp):
            chunk = chunks[i]
            chunk_pixels = pixel_values[chunk['pixel_start']:chunk['pixel_end']]
            chunk_grid = grid_thw[chunk['image_start']:chunk['image_end']]
            vision_refs.append(vision_actors[i].forward.remote(chunk_pixels, chunk_grid))
        
        local_vision_outputs = ray.get(vision_refs)
        vision_end = time.perf_counter()
        
        # === ALL GATHER (vision actors get full embeddings) ===
        all_gather_start = time.perf_counter()
        if vision_dp > 1:
            all_gather_refs = [
                vision_actors[i].all_gather_embeddings.remote(local_vision_outputs[i])
                for i in range(vision_dp)
            ]
            full_vision_outputs = ray.get(all_gather_refs)
        else:
            full_vision_outputs = local_vision_outputs
        all_gather_end = time.perf_counter()
        
        # === LLM FORWARD (TP) - Each LLM actor gets embeddings from colocated vision ===
        llm_start = time.perf_counter()
        llm_refs = [
            llm_actors[i].forward.remote(full_vision_outputs[i])
            for i in range(llm_tp)
        ]
        ray.get(llm_refs)
        llm_end = time.perf_counter()
        
        batch_end = time.perf_counter()
        
        if not is_warmup:
            vision_times.append(vision_end - vision_start)
            all_gather_times.append(all_gather_end - all_gather_start)
            llm_times.append(llm_end - llm_start)
            total_times.append(batch_end - batch_start)
        
        if (batch_idx + 1) % 100 == 0:
            status = "warmup" if is_warmup else "measured"
            logger.info(f"  Batch {batch_idx + 1}/{total_batches} ({status})")
    
    # Cleanup - only vision actors have distributed initialized
    if vision_dp > 1:
        cleanup_refs = [actor.cleanup_distributed.remote() for actor in vision_actors]
        ray.get(cleanup_refs)
    
    for actor in vision_actors + llm_actors:
        ray.kill(actor)
    time.sleep(1)
    
    # Compute statistics
    def stats(times):
        t = torch.tensor(times)
        return {
            'mean': t.mean().item(),
            'std': t.std().item(),
            'min': t.min().item(),
            'max': t.max().item(),
            'p50': t.median().item(),
            'p99': t.quantile(0.99).item(),
        }
    
    results = {
        'mode': 'colocated',
        'vision_dp': vision_dp,
        'llm_tp': llm_tp,
        'total_gpus': total_gpus,
        'num_batches': num_batches,
        'images_per_batch': images_per_batch,
        'vision': stats(vision_times),
        'all_gather': stats(all_gather_times),
        'transfer': {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'p50': 0, 'p99': 0},
        'llm': stats(llm_times),
        'total': stats(total_times),
    }
    
    # Throughput calculations
    total_images = num_batches * images_per_batch
    total_time = sum(total_times)
    total_vision_time = sum(vision_times)
    total_llm_time = sum(llm_times)
    num_visual_tokens_per_batch = visual_tokens_per_image * images_per_batch
    total_visual_tokens = num_batches * num_visual_tokens_per_batch
    
    results['throughput'] = {
        # Legacy compatibility
        'images_per_sec': total_images / total_time,
        'batches_per_sec': num_batches / total_time,
        
        # Vision encoder metrics
        'vision_requests_per_sec': num_batches / total_vision_time,
        'vision_images_per_sec': total_images / total_vision_time,
        'vision_tokens_per_sec': total_visual_tokens / total_vision_time,
        
        # LLM decoder metrics
        'llm_requests_per_sec': num_batches / total_llm_time,
        'llm_images_per_sec': total_images / total_llm_time,
        'llm_tokens_per_sec': total_visual_tokens / total_llm_time,
        
        # End-to-end metrics
        'e2e_requests_per_sec': num_batches / total_time,
        'e2e_images_per_sec': total_images / total_time,
        'e2e_tokens_per_sec': total_visual_tokens / total_time,
    }
    
    results['tokens'] = {
        'visual_tokens_per_image': visual_tokens_per_image,
        'visual_tokens_per_batch': num_visual_tokens_per_batch,
    }
    
    return results


def benchmark_combined(
    model_name: str,
    num_replicas: int,
    num_batches: int,
    images_per_batch: int,
    warmup_batches: int = 10,
    master_port: int = 29500,
) -> Dict[str, Any]:
    """
    Benchmark the combined (full VLM) model.
    
    Uses QwenVLCombinedActor which runs both vision encoder and LLM
    in a single forward pass. This is the baseline for comparison.
    
    Args:
        model_name: Model name or path
        num_replicas: Number of model replicas (data parallel)
        num_batches: Number of batches to benchmark
        images_per_batch: Images per batch per replica
        warmup_batches: Number of warmup batches
        master_port: NCCL master port
    
    Returns:
        Dict with benchmark results
    """
    from qwen_vl_combined_actor import create_combined_actor_class
    
    total_gpus = num_replicas
    
    logger.info(f"\n{'='*60}")
    logger.info(f"COMBINED: {num_replicas} replica(s) of full VLM")
    logger.info(f"  GPUs: 0 to {num_replicas - 1}")
    logger.info(f"  Total GPUs: {total_gpus}")
    logger.info(f"  Batches: {num_batches} (warmup: {warmup_batches})")
    logger.info(f"  Images per batch per replica: {images_per_batch}")
    logger.info(f"{'='*60}")
    
    # Create combined actors
    CombinedActorClass = create_combined_actor_class(num_gpus=1, num_cpus=4)
    actors = [
        CombinedActorClass.remote(
            model_name,
            rank=i,
            world_size=num_replicas,
        )
        for i in range(num_replicas)
    ]
    
    # Initialize distributed if multiple replicas
    if num_replicas > 1:
        master_addr = ray.get(actors[0].get_ip_address.remote())
        init_refs = [
            actor.init_distributed.remote(master_addr, master_port)
            for actor in actors
        ]
        ray.get(init_refs)
        logger.info(f"Distributed initialized (world_size={num_replicas})")
    
    # Build models
    ray.get([actor.build_model.remote() for actor in actors])
    logger.info(f"Models built on {num_replicas} GPU(s)")
    
    # Create batch (same for all replicas if DP > 1)
    pixel_values, grid_thw, visual_tokens_per_image = create_dummy_batch(images_per_batch)
    
    # Timing
    forward_times = []
    
    total_batches_run = warmup_batches + num_batches
    logger.info(f"Starting benchmark ({warmup_batches} warmup + {num_batches} measured)...")
    
    for batch_idx in range(total_batches_run):
        is_warmup = batch_idx < warmup_batches
        
        # === SINGLE FORWARD PASS (vision + LLM combined) ===
        forward_start = time.perf_counter()
        
        forward_refs = [actor.forward.remote(pixel_values, grid_thw) for actor in actors]
        ray.get(forward_refs)
        
        forward_end = time.perf_counter()
        
        if not is_warmup:
            forward_times.append(forward_end - forward_start)
        
        if (batch_idx + 1) % 100 == 0:
            status = "warmup" if is_warmup else "measured"
            logger.info(f"  Batch {batch_idx + 1}/{total_batches_run} ({status})")
    
    # Cleanup
    if num_replicas > 1:
        cleanup_refs = [actor.cleanup_distributed.remote() for actor in actors]
        ray.get(cleanup_refs)
    
    for actor in actors:
        ray.kill(actor)
    time.sleep(1)
    
    # Compute statistics
    def stats(times):
        t = torch.tensor(times)
        return {
            'mean': t.mean().item(),
            'std': t.std().item(),
            'min': t.min().item(),
            'max': t.max().item(),
            'p50': t.median().item(),
            'p99': t.quantile(0.99).item(),
        }
    
    # Total images per batch across all replicas
    total_images_per_batch = images_per_batch * num_replicas
    
    results = {
        'mode': 'combined',
        'vision_dp': num_replicas,
        'llm_tp': 1,
        'total_gpus': total_gpus,
        'num_batches': num_batches,
        'images_per_batch': total_images_per_batch,
        'num_replicas': num_replicas,
        'forward': stats(forward_times),
        # For compatibility with print functions
        'vision': stats(forward_times),  # Combined, so vision = total
        'all_gather': {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'p50': 0, 'p99': 0},
        'transfer': {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'p50': 0, 'p99': 0},
        'llm': {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'p50': 0, 'p99': 0},
        'total': stats(forward_times),
    }
    
    # Throughput calculations (combined: vision + LLM in single pass)
    total_images = num_batches * total_images_per_batch
    total_time = sum(forward_times)
    num_visual_tokens_per_batch = visual_tokens_per_image * total_images_per_batch
    total_visual_tokens = num_batches * num_visual_tokens_per_batch
    
    results['throughput'] = {
        # Legacy compatibility
        'images_per_sec': total_images / total_time,
        'batches_per_sec': num_batches / total_time,
        
        # Vision encoder metrics (same as E2E for combined mode)
        'vision_requests_per_sec': num_batches / total_time,
        'vision_images_per_sec': total_images / total_time,
        'vision_tokens_per_sec': total_visual_tokens / total_time,
        
        # LLM decoder metrics (same as E2E for combined mode)
        'llm_requests_per_sec': num_batches / total_time,
        'llm_images_per_sec': total_images / total_time,
        'llm_tokens_per_sec': total_visual_tokens / total_time,
        
        # End-to-end metrics
        'e2e_requests_per_sec': num_batches / total_time,
        'e2e_images_per_sec': total_images / total_time,
        'e2e_tokens_per_sec': total_visual_tokens / total_time,
    }
    
    results['tokens'] = {
        'visual_tokens_per_image': visual_tokens_per_image,
        'visual_tokens_per_batch': num_visual_tokens_per_batch,
    }
    
    return results


def print_results(results: Dict[str, Any]):
    """Print benchmark results with detailed metrics."""
    mode = results.get('mode', 'unknown')
    print(f"\n{'='*70}")
    print(f"Results: {mode.upper()} (DP={results['vision_dp']}, TP={results['llm_tp']}, {results['total_gpus']} GPUs)")
    print(f"{'='*70}")
    print(f"  Batches: {results['num_batches']}, Images/batch: {results['images_per_batch']}")
    tokens_info = results.get('tokens', {})
    if tokens_info:
        print(f"  Visual tokens/image: {tokens_info.get('visual_tokens_per_image', 'N/A')}")
    print()
    
    # Phase Timings
    print("  LATENCY (seconds per batch):")
    print(f"  {'Phase':<15} {'Mean':>10} {'Std':>10} {'P50':>10} {'P99':>10}")
    print(f"  {'-'*55}")
    
    phases = ['vision', 'all_gather', 'transfer', 'llm', 'total']
    for phase in phases:
        if phase in results and results[phase]['mean'] > 0:
            s = results[phase]
            print(f"  {phase:<15} {s['mean']:>10.4f} {s['std']:>10.4f} {s['p50']:>10.4f} {s['p99']:>10.4f}")
    
    print()
    
    # Detailed throughput by component
    t = results.get('throughput', {})
    
    print("  VISION ENCODER THROUGHPUT:")
    print(f"    Requests/sec:   {t.get('vision_requests_per_sec', t.get('batches_per_sec', 0)):.2f}")
    print(f"    Images/sec:     {t.get('vision_images_per_sec', t.get('images_per_sec', 0)):.2f}")
    print(f"    Tokens/sec:     {t.get('vision_tokens_per_sec', 0):.2f}")
    print()
    
    if mode != 'combined':
        print("  LLM DECODER THROUGHPUT:")
        print(f"    Requests/sec:   {t.get('llm_requests_per_sec', t.get('batches_per_sec', 0)):.2f}")
        print(f"    Images/sec:     {t.get('llm_images_per_sec', t.get('images_per_sec', 0)):.2f}")
        print(f"    Tokens/sec:     {t.get('llm_tokens_per_sec', 0):.2f}")
        print()
    
    print("  END-TO-END THROUGHPUT:")
    print(f"    Requests/sec:   {t.get('e2e_requests_per_sec', t.get('batches_per_sec', 0)):.2f}")
    print(f"    Images/sec:     {t.get('e2e_images_per_sec', t.get('images_per_sec', 0)):.2f}")
    print(f"    Tokens/sec:     {t.get('e2e_tokens_per_sec', 0):.2f}")
    print()


def print_comparison(all_results):
    """Print comparison table."""
    print(f"\n{'='*100}")
    print("COMPARISON TABLE")
    print(f"{'='*100}")
    
    # Latency comparison
    print(f"\n  LATENCY (seconds per batch):")
    print(f"  {'Mode':<12} {'DP':>4} {'TP':>4} {'GPUs':>5} {'Vision':>10} {'LLM':>10} {'Total':>10}")
    print(f"  {'-'*65}")
    
    for r in all_results:
        mode = r['mode'][:10]
        dp = r['vision_dp']
        tp = r['llm_tp']
        gpus = r['total_gpus']
        vision = r['vision']['mean']
        llm = r['llm']['mean']
        total = r['total']['mean']
        print(f"  {mode:<12} {dp:>4} {tp:>4} {gpus:>5} {vision:>10.4f} {llm:>10.4f} {total:>10.4f}")
    
    # Throughput comparison
    print(f"\n  THROUGHPUT:")
    print(f"  {'Mode':<12} {'DP':>4} {'TP':>4} {'Vis img/s':>10} {'Vis tok/s':>10} {'LLM tok/s':>10} {'E2E img/s':>10}")
    print(f"  {'-'*72}")
    
    for r in all_results:
        mode = r['mode'][:10]
        dp = r['vision_dp']
        tp = r['llm_tp']
        t = r['throughput']
        vis_imgs = t.get('vision_images_per_sec', t.get('images_per_sec', 0))
        vis_toks = t.get('vision_tokens_per_sec', 0)
        llm_toks = t.get('llm_tokens_per_sec', 0)
        e2e_imgs = t.get('e2e_images_per_sec', t.get('images_per_sec', 0))
        print(f"  {mode:<12} {dp:>4} {tp:>4} {vis_imgs:>10.2f} {vis_toks:>10.0f} {llm_toks:>10.0f} {e2e_imgs:>10.2f}")
    
    print()


def main():
    parser = argparse.ArgumentParser(description="Vision DP + LLM TP Benchmark")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="Model name or path",
    )
    parser.add_argument(
        "--vision_dp",
        type=int,
        default=2,
        help="Vision data parallel size (default: 2)",
    )
    parser.add_argument(
        "--llm_tp",
        type=int,
        default=2,
        help="LLM tensor parallel size (default: 2)",
    )
    parser.add_argument(
        "--colocated",
        action="store_true",
        help="Colocated mode: vision and LLM share same GPUs (requires DP == TP)",
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=50,
        help="Number of batches to benchmark (default: 50)",
    )
    parser.add_argument(
        "--images_per_batch",
        type=int,
        default=10,
        help="Number of images per batch (default: 10)",
    )
    parser.add_argument(
        "--warmup_batches",
        type=int,
        default=10,
        help="Number of warmup batches (default: 10)",
    )
    parser.add_argument(
        "--master_port",
        type=int,
        default=29500,
        help="NCCL master port (default: 29500)",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Combined mode: run full VLM (vision+LLM) in single actor",
    )
    parser.add_argument(
        "--num_replicas",
        type=int,
        default=1,
        help="Number of replicas for combined mode (default: 1)",
    )
    
    args = parser.parse_args()
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init()
    
    # Validate arguments
    if args.combined and args.colocated:
        print("ERROR: Cannot use --combined and --colocated together")
        return 1
    
    if args.colocated and args.vision_dp != args.llm_tp:
        print(f"ERROR: Colocated mode requires DP == TP, got DP={args.vision_dp}, TP={args.llm_tp}")
        return 1
    
    if args.combined:
        mode_str = "COMBINED"
        total_gpus = args.num_replicas
    elif args.colocated:
        mode_str = "COLOCATED"
        total_gpus = args.vision_dp
    else:
        mode_str = "NON-COLOCATED"
        total_gpus = args.vision_dp + args.llm_tp
    
    print("="*70)
    print("Vision DP + LLM TP Benchmark")
    print("="*70)
    print(f"Model: {args.model_name}")
    print(f"Mode: {mode_str}")
    if args.combined:
        print(f"Replicas: {args.num_replicas}")
    else:
        print(f"Vision DP: {args.vision_dp}")
        print(f"LLM TP: {args.llm_tp}")
    print(f"Total GPUs: {total_gpus}")
    print(f"Batches: {args.num_batches} (warmup: {args.warmup_batches})")
    print(f"Images per batch: {args.images_per_batch}")
    print("="*70)
    
    # Run benchmark
    if args.combined:
        results = benchmark_combined(
            model_name=args.model_name,
            num_replicas=args.num_replicas,
            num_batches=args.num_batches,
            images_per_batch=args.images_per_batch,
            warmup_batches=args.warmup_batches,
            master_port=args.master_port,
        )
    elif args.colocated:
        results = benchmark_colocated(
            model_name=args.model_name,
            dp_tp_size=args.vision_dp,
            num_batches=args.num_batches,
            images_per_batch=args.images_per_batch,
            warmup_batches=args.warmup_batches,
            master_port=args.master_port,
        )
    else:
        results = benchmark_non_colocated(
            model_name=args.model_name,
            vision_dp=args.vision_dp,
            llm_tp=args.llm_tp,
            num_batches=args.num_batches,
            images_per_batch=args.images_per_batch,
            warmup_batches=args.warmup_batches,
            master_port=args.master_port,
        )
    
    print_results(results)
    
    print("="*70)
    print("Benchmark complete!")
    print("="*70)
    
    return 0


if __name__ == "__main__":
    exit(main())
