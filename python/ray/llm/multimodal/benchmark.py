#!/usr/bin/env python3
"""
Benchmark script for comparing encoder vs decoder throughput
with different DP/TP configurations.

Uses the same NCCL P2P transfer approach as verify_equivalence.py.

Measures:
- Vision encoder throughput (images/sec)
- LLM decoder throughput (tokens/sec)
- End-to-end throughput

Communication:
- NCCL all_gather for vision embeddings (among vision actors)
- NCCL P2P send/recv for vision-to-LLM transfer
"""

import argparse
import logging
import time
from typing import Tuple, List, Dict, Any

import ray
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_dummy_batch(
    num_images: int,
    image_height: int = 448,  # 252 = 14 * 18, divisible by patch_size
    image_width: int = 448,   # 252 = 14 * 18, divisible by patch_size
    patch_size: int = 14,
    temporal_patch_size: int = 2,
    in_channels: int = 3,
    spatial_merge_size: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Create a dummy batch of images for benchmarking.
    Matches verify_equivalence.py format.
    
    For Qwen3-VL with 252x252 images:
    - grid_h = grid_w = 252 // 14 = 18 patches
    - After spatial merge (2x2): 18/2 * 18/2 = 81 visual tokens per image
    
    Returns:
        pixel_values: [total_patches, patch_dim]
        grid_thw: [num_images, 3]
        visual_tokens_per_image: number of visual tokens output per image
    """
    h_patches = image_height // patch_size
    w_patches = image_width // patch_size
    t_patches = 1  # Single frame for images
    
    patches_per_image = t_patches * h_patches * w_patches
    total_patches = num_images * patches_per_image
    
    # Visual tokens after spatial merge (merge_size x merge_size patches -> 1 token)
    visual_tokens_per_image = (t_patches * h_patches * w_patches) // (spatial_merge_size ** 2)
    
    # Patch dimension: temporal_patch_size * in_channels * patch_size * patch_size
    patch_dim = temporal_patch_size * in_channels * patch_size * patch_size
    
    # Create pixel values [total_patches, patch_dim]
    torch.manual_seed(42)
    pixel_values = torch.randn(total_patches, patch_dim, dtype=torch.float32)
    
    # Create grid_thw [num_images, 3]
    grid_thw = torch.tensor([[t_patches, h_patches, w_patches]] * num_images, dtype=torch.long)
    
    return pixel_values, grid_thw, visual_tokens_per_image


def benchmark_config(
    model_name: str,
    vision_dp: int,
    llm_tp: int,
    num_batches: int,
    images_per_batch: int,
    warmup_batches: int = 10,
    master_port: int = 29510,
) -> Dict[str, Any]:
    """
    Benchmark a specific DP/TP configuration.
    
    Uses the same flow as verify_equivalence.py:
    - All actors in same NCCL world (vision_dp + llm_tp)
    - NCCL all_gather among vision actors
    - NCCL P2P send/recv from vision actor 0 to all LLM actors
    
    Returns dict with timing statistics.
    """
    from qwen3_vl_vision_actor import create_vision_actor_class
    from qwen3_vl_llm_actor import create_llm_actor_class
    
    total_world_size = vision_dp + llm_tp
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Benchmarking: Vision DP={vision_dp}, LLM TP={llm_tp}, images_per_batch={images_per_batch}")
    logger.info(f"  Vision GPUs: 0 to {vision_dp - 1}")
    logger.info(f"  LLM GPUs: {vision_dp} to {vision_dp + llm_tp - 1}")
    logger.info(f"  Total world size: {total_world_size}")
    logger.info(f"  Batches: {num_batches} (warmup: {warmup_batches})")
    logger.info(f"  Images per batch: {images_per_batch}")
    logger.info(f"  Using NCCL all_gather for vision embeddings")
    logger.info(f"  Using NCCL P2P for vision-to-LLM transfer")
    logger.info(f"{'='*60}")
    
    # Create vision actors
    VisionActorClass = create_vision_actor_class(num_gpus=1, num_cpus=4)
    vision_actors = [
        VisionActorClass.remote(model_name, rank=i, dp_size=vision_dp)
        for i in range(vision_dp)
    ]
    
    # Create LLM actors with global ranks (matching verify_equivalence.py)
    LLMActorClass = create_llm_actor_class(num_gpus=1, num_cpus=4)
    llm_actors = [
        LLMActorClass.remote(
            model_name,
            tp_rank=i,
            tp_size=llm_tp,
            global_rank=vision_dp + i,  # Global rank: vision_dp, vision_dp+1, ...
        )
        for i in range(llm_tp)
    ]
    
    # Get master address from vision rank 0
    master_addr = ray.get(vision_actors[0].get_ip_address.remote())
    
    # Initialize distributed for ALL actors in the same world (like verify_equivalence.py)
    vision_init_refs = [
        actor.init_distributed.remote(
            master_addr, master_port,
            world_size=total_world_size,
            llm_tp_size=llm_tp,
        )
        for actor in vision_actors
    ]
    
    llm_init_refs = [
        actor.init_distributed.remote(
            master_addr, master_port,
            world_size=total_world_size,
            vision_dp_size=vision_dp,
        )
        for actor in llm_actors
    ]
    
    # Initialize all actors in the same process group
    ray.get(vision_init_refs + llm_init_refs)
    logger.info(f"Distributed initialized (world_size={total_world_size})")
    
    # Build models
    vision_build_refs = [actor.build_model.remote() for actor in vision_actors]
    llm_build_refs = [actor.build_model.remote() for actor in llm_actors]
    ray.get(vision_build_refs + llm_build_refs)
    logger.info(f"All models built")
    
    # Create a single batch for benchmarking (reused)
    # 252x252 images -> 81 visual tokens per image after spatial merge
    pixel_values, grid_thw, visual_tokens_per_image = create_dummy_batch(images_per_batch)
    
    # Calculate batch splitting info
    num_images = grid_thw.shape[0]
    images_per_actor = num_images // vision_dp
    extra = num_images % vision_dp
    patches_per_image = grid_thw[0, 0].item() * grid_thw[0, 1].item() * grid_thw[0, 2].item()
    
    # Pre-compute chunk boundaries
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
    
    # Timing accumulators
    vision_times = []
    all_gather_times = []
    p2p_times = []
    llm_times = []
    total_times = []
    
    total_batches = warmup_batches + num_batches
    
    logger.info(f"Starting benchmark ({warmup_batches} warmup + {num_batches} measured)...")
    
    for batch_idx in range(total_batches):
        is_warmup = batch_idx < warmup_batches
        
        batch_start = time.perf_counter()
        
        # === VISION FORWARD ===
        vision_start = time.perf_counter()
        
        vision_refs = []
        for i in range(vision_dp):
            chunk = chunks[i]
            chunk_pixels = pixel_values[chunk['pixel_start']:chunk['pixel_end']]
            chunk_grid = grid_thw[chunk['image_start']:chunk['image_end']]
            vision_refs.append(vision_actors[i].forward.remote(chunk_pixels, chunk_grid))
        
        local_vision_outputs = ray.get(vision_refs)
        vision_end = time.perf_counter()
        
        # === ALL GATHER ===
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
        
        # === P2P TRANSFER (NCCL send/recv like verify_equivalence.py) ===
        p2p_start = time.perf_counter()
        
        embedding_shape = (vision_embeddings.shape[0], vision_embeddings.shape[1])
        
        # Vision actor 0 sends to each LLM actor, LLM actors receive simultaneously
        send_refs = []
        recv_refs = []
        
        for i in range(llm_tp):
            dst_rank = vision_dp + i  # LLM global ranks start at vision_dp
            send_refs.append(vision_actors[0].send_embeddings_p2p.remote(vision_embeddings, dst_rank))
            recv_refs.append(llm_actors[i].recv_embeddings_p2p.remote(embedding_shape, src_rank=0))
        
        # Wait for P2P transfers to complete and get received embeddings
        ray.get(send_refs)
        received_embeddings = ray.get(recv_refs)
        
        p2p_end = time.perf_counter()
        
        # === LLM FORWARD ===
        llm_start = time.perf_counter()
        
        llm_refs = [
            actor.forward.remote(received_embeddings[i])
            for i, actor in enumerate(llm_actors)
        ]
        llm_outputs = ray.get(llm_refs)
        
        llm_end = time.perf_counter()
        
        batch_end = time.perf_counter()
        
        # Record times (skip warmup)
        if not is_warmup:
            vision_times.append(vision_end - vision_start)
            all_gather_times.append(all_gather_end - all_gather_start)
            p2p_times.append(p2p_end - p2p_start)
            llm_times.append(llm_end - llm_start)
            total_times.append(batch_end - batch_start)
        
        if (batch_idx + 1) % 100 == 0:
            status = "warmup" if is_warmup else "measured"
            logger.info(f"  Batch {batch_idx + 1}/{total_batches} ({status})")
    
    # Cleanup distributed before killing actors (matching verify_equivalence.py)
    cleanup_refs = []
    for actor in vision_actors:
        cleanup_refs.append(actor.cleanup_distributed.remote())
    for actor in llm_actors:
        cleanup_refs.append(actor.cleanup_distributed.remote())
    ray.get(cleanup_refs)
    
    for actor in vision_actors:
        ray.kill(actor)
    for actor in llm_actors:
        ray.kill(actor)
    
    time.sleep(1)  # Allow NCCL resources to be released
    
    # Compute statistics
    def stats(times):
        times = torch.tensor(times)
        return {
            'mean': times.mean().item(),
            'std': times.std().item(),
            'min': times.min().item(),
            'max': times.max().item(),
            'p50': times.median().item(),
            'p99': times.quantile(0.99).item(),
        }
    
    results = {
        'config': f'DP={vision_dp}, TP={llm_tp}',
        'vision_dp': vision_dp,
        'llm_tp': llm_tp,
        'total_gpus': vision_dp + llm_tp,
        'num_batches': num_batches,
        'images_per_batch': images_per_batch,
        'vision': stats(vision_times),
        'all_gather': stats(all_gather_times),
        'p2p': stats(p2p_times),
        'llm': stats(llm_times),
        'total': stats(total_times),
    }
    
    # Compute throughput
    total_images = num_batches * images_per_batch
    total_time = sum(total_times)
    
    # Visual tokens: calculated from image size (252x252 -> 81 tokens per image)
    num_visual_tokens_per_batch = visual_tokens_per_image * images_per_batch
    total_visual_tokens = num_batches * num_visual_tokens_per_batch
    
    # LLM output: for a single forward pass, output tokens = input tokens (hidden states)
    # In real generation, this would be different (autoregressive)
    llm_output_tokens_per_batch = num_visual_tokens_per_batch  # Same shape in/out for 1 forward
    
    results['throughput'] = {
        # E2E throughput
        'images_per_sec': total_images / total_time,
        'batches_per_sec': num_batches / total_time,  # req/s
        # Vision throughput (input: requests, output: visual tokens)
        'vision_req_per_sec': num_batches / sum(vision_times),  # req/s (1 req = 1 batch)
        'vision_tokens_per_sec': total_visual_tokens / sum(vision_times),  # visual tokens/s
        # LLM throughput (input: tokens, output: tokens)
        'llm_input_tokens_per_sec': total_visual_tokens / sum(llm_times),  # input tokens/s
        'llm_output_tokens_per_sec': total_visual_tokens / sum(llm_times),  # output tokens/s (same for 1 fwd pass)
    }
    
    # Store token counts for reference
    results['tokens'] = {
        'visual_tokens_per_image': visual_tokens_per_image,
        'visual_tokens_per_batch': num_visual_tokens_per_batch,
        'total_visual_tokens': total_visual_tokens,
        'image_size': '252x252',
    }
    
    return results


def benchmark_combined(
    model_name: str,
    num_batches: int,
    images_per_batch: int,
    num_replicas: int = 1,
    tp_size: int = 1,
    warmup_batches: int = 10,
    master_port: int = 29600,
) -> Dict[str, Any]:
    """
    Benchmark combined models (vision + LLM in single actor).
    
    Uses a SINGLE forward pass that wraps both vision encoder and LLM,
    measuring true end-to-end throughput without artificial phase separation.
    
    Args:
        model_name: Model name or path
        num_batches: Number of batches to benchmark
        images_per_batch: Images per batch per replica
        num_replicas: Number of model replicas (data parallel, each with tp_size GPUs)
        tp_size: Tensor parallel size per replica (GPUs per replica)
        warmup_batches: Number of warmup batches
        master_port: NCCL master port
    
    With num_replicas=1, tp_size=1: 1 GPU, single model
    With num_replicas=1, tp_size=2: 2 GPUs, single model with TP
    With num_replicas=N, tp_size=1: N GPUs, N independent models (DP)
    
    Returns dict with timing statistics.
    """
    from qwen_vl_combined_actor import create_combined_actor_class
    
    total_gpus = num_replicas * tp_size
    
    logger.info(f"Benchmarking COMBINED model (single forward pass):")
    logger.info(f"  Replicas: {num_replicas}, TP size: {tp_size}, Total GPUs: {total_gpus}")
    logger.info(f"  Batches: {num_batches} (+ {warmup_batches} warmup)")
    logger.info(f"  Images per batch per replica: {images_per_batch}")
    logger.info(f"  Total images per batch: {images_per_batch * num_replicas}")
    
    # Create combined actors
    # For TP > 1, each replica has tp_size actors that form a TP group
    CombinedActorClass = create_combined_actor_class(num_gpus=1, num_cpus=4)
    
    actors = []
    for replica_id in range(num_replicas):
        if tp_size == 1:
            # Single GPU per replica - no distributed needed
            actor = CombinedActorClass.remote(model_name, rank=0, tp_size=1)
            actors.append([actor])
        else:
            # Multiple GPUs per replica - TP within replica
            replica_actors = []
            for tp_rank in range(tp_size):
                actor = CombinedActorClass.remote(
                    model_name, 
                    rank=tp_rank, 
                    tp_size=tp_size,
                    master_port=master_port + replica_id * 100  # Different port per replica
                )
                replica_actors.append(actor)
            actors.append(replica_actors)
    
    # Initialize distributed for TP > 1
    if tp_size > 1:
        for replica_id, replica_actors in enumerate(actors):
            # Get master address from rank 0 of this replica
            master_addr = ray.get(replica_actors[0].get_ip_address.remote())
            port = master_port + replica_id * 100
            
            init_refs = [
                actor.init_distributed.remote(master_addr, port)
                for actor in replica_actors
            ]
            ray.get(init_refs)
        logger.info(f"Distributed initialized for TP={tp_size}")
    
    # Build the models
    build_refs = []
    for replica_actors in actors:
        for actor in replica_actors:
            build_refs.append(actor.build_model.remote())
    ray.get(build_refs)
    logger.info(f"Combined model(s) built on {total_gpus} GPU(s)")
    
    # Create batch (reused for all iterations)
    pixel_values, grid_thw, visual_tokens_per_image = create_dummy_batch(images_per_batch)
    logger.info(f"Batch created: {images_per_batch} images, {visual_tokens_per_image} visual tokens/image")
    
    # Timing array (single forward, no phase separation)
    forward_times = []
    
    total_batches = warmup_batches + num_batches
    
    for batch_idx in range(total_batches):
        is_warmup = batch_idx < warmup_batches
        
        # === SINGLE FORWARD PASS (vision + LLM combined) ===
        forward_start = time.perf_counter()
        
        forward_refs = []
        for replica_actors in actors:
            if tp_size == 1:
                # Single actor per replica
                forward_refs.append(replica_actors[0].forward.remote(pixel_values, grid_thw))
            else:
                # For TP, only rank 0 returns output (others participate in distributed ops)
                # All actors run forward, but we only collect from rank 0
                for actor in replica_actors:
                    forward_refs.append(actor.forward.remote(pixel_values, grid_thw))
        
        outputs = ray.get(forward_refs)
        forward_end = time.perf_counter()
        
        # Record times (skip warmup)
        if not is_warmup:
            forward_times.append(forward_end - forward_start)
        
        if (batch_idx + 1) % 100 == 0:
            status = "warmup" if is_warmup else "measured"
            logger.info(f"  Batch {batch_idx + 1}/{total_batches} ({status})")
    
    # Cleanup
    if tp_size > 1:
        cleanup_refs = []
        for replica_actors in actors:
            for actor in replica_actors:
                cleanup_refs.append(actor.cleanup_distributed.remote())
        ray.get(cleanup_refs)
    
    for replica_actors in actors:
        for actor in replica_actors:
            ray.kill(actor)
    time.sleep(1)
    
    # Compute statistics
    def stats(times):
        times = torch.tensor(times)
        return {
            'mean': times.mean().item(),
            'std': times.std().item(),
            'min': times.min().item(),
            'max': times.max().item(),
            'p50': times.median().item(),
            'p99': times.quantile(0.99).item(),
        }
    
    # Total images per batch across all replicas
    total_images_per_batch = images_per_batch * num_replicas
    
    results = {
        'config': f'Combined (DP={num_replicas}, TP={tp_size})',
        'vision_dp': num_replicas,
        'llm_tp': tp_size,
        'total_gpus': total_gpus,
        'num_batches': num_batches,
        'images_per_batch': total_images_per_batch,
        'num_replicas': num_replicas,
        'tp_size': tp_size,
        'forward': stats(forward_times),
        # For compatibility with print functions
        'vision': {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'p50': 0, 'p99': 0},
        'all_gather': {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'p50': 0, 'p99': 0},
        'p2p': {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'p50': 0, 'p99': 0},
        'llm': {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'p50': 0, 'p99': 0},
        'total': stats(forward_times),  # total = forward for combined
    }
    
    # Compute throughput
    total_images = num_batches * total_images_per_batch
    total_time = sum(forward_times)
    
    # Visual tokens per batch (across all replicas)
    num_visual_tokens_per_batch = visual_tokens_per_image * total_images_per_batch
    total_visual_tokens = num_batches * num_visual_tokens_per_batch
    
    results['throughput'] = {
        # E2E throughput (total across all replicas)
        'images_per_sec': total_images / total_time,
        'batches_per_sec': num_batches / total_time,
        # Combined forward (no separate vision/LLM)
        'forward_tokens_per_sec': total_visual_tokens / total_time,
        # For compatibility
        'vision_req_per_sec': num_batches / total_time,
        'vision_tokens_per_sec': total_visual_tokens / total_time,
        'llm_input_tokens_per_sec': total_visual_tokens / total_time,
        'llm_output_tokens_per_sec': total_visual_tokens / total_time,
    }
    
    results['tokens'] = {
        'visual_tokens_per_image': visual_tokens_per_image,
        'visual_tokens_per_batch': num_visual_tokens_per_batch,
        'total_visual_tokens': total_visual_tokens,
        'image_size': f'{448}x{448}',  # Using default from create_dummy_batch
    }
    
    return results


def print_combined_results(results: Dict[str, Any]):
    """Print combined benchmark results."""
    num_replicas = results.get('num_replicas', 1)
    tp_size = results.get('tp_size', 1)
    print(f"\n{'='*70}")
    print(f"COMBINED MODEL BENCHMARK (DP={num_replicas}, TP={tp_size}, {results['total_gpus']} GPU(s))")
    print(f"{'='*70}")
    print(f"  Batches: {results['num_batches']}, Total Images/batch: {results['images_per_batch']}")
    if num_replicas > 1:
        print(f"  Images/batch/replica: {results['images_per_batch'] // num_replicas}")
    print()
    
    print("  Forward Timing (single pass, vision+LLM combined):")
    print(f"  {'Phase':<15} {'Mean':>10} {'Std':>10} {'P50':>10} {'P99':>10}")
    print(f"  {'-'*55}")
    
    if 'forward' in results:
        s = results['forward']
        print(f"  {'forward':<15} {s['mean']:>10.4f} {s['std']:>10.4f} {s['p50']:>10.4f} {s['p99']:>10.4f}")
    else:
        # Fallback for old format
        for phase in ['vision', 'llm', 'total']:
            s = results[phase]
            print(f"  {phase:<15} {s['mean']:>10.4f} {s['std']:>10.4f} {s['p50']:>10.4f} {s['p99']:>10.4f}")
    
    print()
    
    # Token info
    tok = results['tokens']
    print(f"  Image size: {tok['image_size']}")
    print(f"  Visual Tokens: {tok['visual_tokens_per_batch']} per batch, {tok['visual_tokens_per_image']} per image")
    print()
    
    print("  Throughput (end-to-end, single forward pass):")
    t = results['throughput']
    print(f"    Images/sec:         {t['images_per_sec']:.2f}")
    print(f"    Requests/sec:       {t['batches_per_sec']:.2f}")
    print(f"    Tokens/sec:         {t.get('forward_tokens_per_sec', t['vision_tokens_per_sec']):.2f}")
    print()


def print_results(results: Dict[str, Any]):
    """Print benchmark results in a formatted way."""
    print(f"\n{'='*70}")
    print(f"Results for {results['config']} ({results['total_gpus']} GPUs, {results['images_per_batch']} img/batch)")
    print(f"{'='*70}")
    print(f"  Batches: {results['num_batches']}, Images/batch: {results['images_per_batch']}")
    print()
    
    print("  Phase Timings (seconds per batch):")
    print(f"  {'Phase':<15} {'Mean':>10} {'Std':>10} {'P50':>10} {'P99':>10}")
    print(f"  {'-'*55}")
    
    for phase in ['vision', 'all_gather', 'p2p', 'llm', 'total']:
        s = results[phase]
        print(f"  {phase:<15} {s['mean']:>10.4f} {s['std']:>10.4f} {s['p50']:>10.4f} {s['p99']:>10.4f}")
    
    print()
    
    # Token info
    tok = results['tokens']
    print(f"  Visual Tokens: {tok['visual_tokens_per_batch']} per batch, {tok['visual_tokens_per_image']:.1f} per image")
    print()
    
    print("  Throughput:")
    t = results['throughput']
    print(f"    End-to-end: {t['images_per_sec']:.2f} images/sec, {t['batches_per_sec']:.2f} req/sec")
    print()
    print(f"    Vision encoder:")
    print(f"      Input:  {t['vision_req_per_sec']:.2f} req/sec (1 req = {results['images_per_batch']} images)")
    print(f"      Output: {t['vision_tokens_per_sec']:.2f} visual_tokens/sec")
    print()
    print(f"    LLM decoder:")
    print(f"      Input:  {t['llm_input_tokens_per_sec']:.2f} tokens/sec")
    print(f"      Output: {t['llm_output_tokens_per_sec']:.2f} tokens/sec")
    print()


def print_comparison_table(all_results: List[Dict[str, Any]], title: str = "COMPARISON TABLE"):
    """Print comparison table for multiple configurations."""
    print(f"\n{'='*100}")
    print(title)
    print(f"{'='*100}")
    
    # Header
    print(f"\n  {'Config':<15} {'Batch':>6} {'GPUs':>5} {'Vision(s)':>10} {'LLM(s)':>10} {'Total(s)':>10} {'E2E req/s':>10} {'Vis tok/s':>12} {'LLM tok/s':>12}")
    print(f"  {'-'*105}")
    
    for r in all_results:
        config = r['config']
        batch = r['images_per_batch']
        gpus = r['total_gpus']
        vision = r['vision']['mean']
        llm = r['llm']['mean']
        total = r['total']['mean']
        e2e_req = r['throughput']['batches_per_sec']
        vision_tok = r['throughput']['vision_tokens_per_sec']
        llm_tok = r['throughput']['llm_input_tokens_per_sec']
        print(f"  {config:<15} {batch:>6} {gpus:>5} {vision:>10.4f} {llm:>10.4f} {total:>10.4f} {e2e_req:>10.2f} {vision_tok:>12.2f} {llm_tok:>12.2f}")
    
    print()
    
    # Speedup relative to first config (baseline)
    if len(all_results) > 1:
        baseline = all_results[0]
        print("  Speedup vs baseline:")
        print(f"  {'Config':<15} {'Batch':>6} {'Vision':>10} {'LLM':>10} {'Total':>10} {'Throughput':>12}")
        print(f"  {'-'*70}")
        
        for r in all_results:
            config = r['config']
            batch = r['images_per_batch']
            vision_speedup = baseline['vision']['mean'] / r['vision']['mean']
            llm_speedup = baseline['llm']['mean'] / r['llm']['mean']
            total_speedup = baseline['total']['mean'] / r['total']['mean']
            throughput_speedup = r['throughput']['images_per_sec'] / baseline['throughput']['images_per_sec']
            print(f"  {config:<15} {batch:>6} {vision_speedup:>9.2f}x {llm_speedup:>9.2f}x {total_speedup:>9.2f}x {throughput_speedup:>11.2f}x")
    
    print()
    
    # Time breakdown
    print("  Time Breakdown (% of total):")
    print(f"  {'Config':<15} {'Batch':>6} {'Vision':>10} {'AllGather':>10} {'P2P':>10} {'LLM':>10}")
    print(f"  {'-'*70}")
    
    for r in all_results:
        config = r['config']
        batch = r['images_per_batch']
        total = r['total']['mean']
        vision_pct = r['vision']['mean'] / total * 100
        gather_pct = r['all_gather']['mean'] / total * 100
        p2p_pct = r['p2p']['mean'] / total * 100
        llm_pct = r['llm']['mean'] / total * 100
        print(f"  {config:<15} {batch:>6} {vision_pct:>9.1f}% {gather_pct:>9.1f}% {p2p_pct:>9.1f}% {llm_pct:>9.1f}%")
    
    print()


def run_experiment_1(args, base_port: int) -> List[Dict[str, Any]]:
    """
    Experiment 1: Effect of batch size with TP=4, DP=4
    Compare images_per_batch = [16, 64, 256]
    """
    print("\n" + "="*100)
    print("EXPERIMENT 1: Effect of Batch Size (TP=4, DP=4)")
    print("="*100)
    
    all_results = []
    batch_sizes = [16, 64, 256]
    
    # First run baseline DP=1, TP=1 for each batch size
    for idx, batch_size in enumerate(batch_sizes):
        # Baseline
        baseline = benchmark_config(
            model_name=args.model_name,
            vision_dp=1,
            llm_tp=1,
            num_batches=args.num_batches,
            images_per_batch=batch_size,
            warmup_batches=args.warmup_batches,
            master_port=base_port + idx * 100,
        )
        baseline['config'] = 'DP=1, TP=1'
        print_results(baseline)
        all_results.append(baseline)
        
        time.sleep(10)
        
        # DP=4, TP=4
        result = benchmark_config(
            model_name=args.model_name,
            vision_dp=4,
            llm_tp=4,
            num_batches=args.num_batches,
            images_per_batch=batch_size,
            warmup_batches=args.warmup_batches,
            master_port=base_port + idx * 100 + 50,
        )
        result['config'] = 'DP=4, TP=4'
        print_results(result)
        all_results.append(result)
        
        if idx < len(batch_sizes) - 1:
            time.sleep(10)
    
    print_comparison_table(all_results, "EXPERIMENT 1: Batch Size Effect (DP=4, TP=4 vs Baseline)")
    return all_results


def run_experiment_2(args, base_port: int) -> List[Dict[str, Any]]:
    """
    Experiment 2: DP scaling with images_per_batch=256, TP=2
    DP ranges from [1, 2, 4, 6]
    """
    print("\n" + "="*100)
    print("EXPERIMENT 2: DP Scaling (images_per_batch=256, TP=2)")
    print("="*100)
    
    all_results = []
    dp_values = [1, 2, 4, 6]
    images_per_batch = 256
    
    # First run baseline DP=1, TP=1
    baseline = benchmark_config(
        model_name=args.model_name,
        vision_dp=1,
        llm_tp=1,
        num_batches=args.num_batches,
        images_per_batch=images_per_batch,
        warmup_batches=args.warmup_batches,
        master_port=base_port,
    )
    baseline['config'] = 'DP=1, TP=1'
    print_results(baseline)
    all_results.append(baseline)
    
    time.sleep(10)
    
    # Run DP scaling with TP=2
    for idx, dp in enumerate(dp_values):
        result = benchmark_config(
            model_name=args.model_name,
            vision_dp=dp,
            llm_tp=2,
            num_batches=args.num_batches,
            images_per_batch=images_per_batch,
            warmup_batches=args.warmup_batches,
            master_port=base_port + (idx + 1) * 100,
        )
        result['config'] = f'DP={dp}, TP=2'
        print_results(result)
        all_results.append(result)
        
        if idx < len(dp_values) - 1:
            time.sleep(10)
    
    print_comparison_table(all_results, "EXPERIMENT 2: DP Scaling (256 img/batch, TP=2)")
    return all_results


def run_experiment_3(args, base_port: int) -> List[Dict[str, Any]]:
    """
    Experiment 3: DP scaling with images_per_batch=16, TP=2
    DP ranges from [1, 2, 4, 6]
    """
    print("\n" + "="*100)
    print("EXPERIMENT 3: DP Scaling (images_per_batch=16, TP=2)")
    print("="*100)
    
    all_results = []
    dp_values = [1, 2, 4, 6]
    images_per_batch = 16
    
    # First run baseline DP=1, TP=1
    baseline = benchmark_config(
        model_name=args.model_name,
        vision_dp=1,
        llm_tp=1,
        num_batches=args.num_batches,
        images_per_batch=images_per_batch,
        warmup_batches=args.warmup_batches,
        master_port=base_port,
    )
    baseline['config'] = 'DP=1, TP=1'
    print_results(baseline)
    all_results.append(baseline)
    
    time.sleep(10)
    
    # Run DP scaling with TP=2
    for idx, dp in enumerate(dp_values):
        result = benchmark_config(
            model_name=args.model_name,
            vision_dp=dp,
            llm_tp=2,
            num_batches=args.num_batches,
            images_per_batch=images_per_batch,
            warmup_batches=args.warmup_batches,
            master_port=base_port + (idx + 1) * 100,
        )
        result['config'] = f'DP={dp}, TP=2'
        print_results(result)
        all_results.append(result)
        
        if idx < len(dp_values) - 1:
            time.sleep(10)
    
    print_comparison_table(all_results, "EXPERIMENT 3: DP Scaling (16 img/batch, TP=2)")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Benchmark encoder vs decoder throughput")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="Model name or path",
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=500,
        help="Number of batches to benchmark (default: 500)",
    )
    parser.add_argument(
        "--images_per_batch",
        type=int,
        default=18,
        help="Number of images per batch (default: 18)",
    )
    parser.add_argument(
        "--warmup_batches",
        type=int,
        default=10,
        help="Number of warmup batches (default: 10)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["1", "2", "3", "all"],
        default=None,
        help="Run specific experiment: 1 (batch size), 2 (DP scaling 256), 3 (DP scaling 16), all",
    )
    parser.add_argument(
        "--suite",
        type=str,
        choices=["tp1", "tp2", "all"],
        default=None,
        help="Test suite: tp1 (DP=1,2,3 TP=1), tp2 (DP=1,2 TP=2), all (both suites)",
    )
    parser.add_argument(
        "--vision_dp",
        type=int,
        default=None,
        help="Vision DP size (for custom config, use with --llm_tp)",
    )
    parser.add_argument(
        "--llm_tp",
        type=int,
        default=None,
        help="LLM TP size (for custom config, use with --vision_dp)",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Run combined model benchmark (vision + LLM in single actor, single forward pass)",
    )
    parser.add_argument(
        "--num_replicas",
        type=int,
        default=1,
        help="Number of model replicas for combined benchmark (DP, default: 1)",
    )
    parser.add_argument(
        "--combined_tp",
        type=int,
        default=1,
        help="Tensor parallel size per replica for combined benchmark (default: 1)",
    )
    
    args = parser.parse_args()
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init()
    
    print("="*100)
    print("Encoder vs Decoder Throughput Benchmark")
    print("="*100)
    print(f"Model: {args.model_name}")
    print(f"Batches: {args.num_batches} (warmup: {args.warmup_batches})")
    print()
    
    # Run combined benchmark if specified
    if args.combined:
        total_gpus = args.num_replicas * args.combined_tp
        print(f"Running COMBINED model benchmark (single forward pass)")
        print(f"  Replicas (DP): {args.num_replicas}")
        print(f"  TP per replica: {args.combined_tp}")
        print(f"  Total GPUs: {total_gpus}")
        print(f"  Images per batch per replica: {args.images_per_batch}")
        print(f"  Total images per batch: {args.images_per_batch * args.num_replicas}")
        print("="*100)
        
        results = benchmark_combined(
            model_name=args.model_name,
            num_batches=args.num_batches,
            images_per_batch=args.images_per_batch,
            num_replicas=args.num_replicas,
            tp_size=args.combined_tp,
            warmup_batches=args.warmup_batches,
        )
        print_combined_results(results)
        
        print("="*100)
        print("Combined benchmark complete!")
        print("="*100)
        return
    
    # Run experiments if specified
    if args.experiment:
        if args.experiment == "1" or args.experiment == "all":
            run_experiment_1(args, base_port=29500)
            time.sleep(15)
        
        if args.experiment == "2" or args.experiment == "all":
            run_experiment_2(args, base_port=30500)
            time.sleep(15)
        
        if args.experiment == "3" or args.experiment == "all":
            run_experiment_3(args, base_port=31500)
        
        print("="*100)
        print("All experiments complete!")
        print("="*100)
        return
    
    # Determine which configurations to run
    configs = []
    
    if args.vision_dp is not None and args.llm_tp is not None:
        # Custom config
        configs.append((args.vision_dp, args.llm_tp))
    elif args.suite == "tp1":
        # DP=1,2,3 with TP=1
        configs = [(1, 1), (2, 1), (3, 1)]
    elif args.suite == "tp2":
        # DP=1,2 with TP=2
        configs = [(1, 2), (2, 2)]
    elif args.suite == "all":
        # Both suites
        configs = [(1, 1), (2, 1), (3, 1), (1, 2), (2, 2)]
    else:
        # Default: tp2 suite
        configs = [(1, 2), (2, 2)]
    
    print(f"Images per batch: {args.images_per_batch}")
    print(f"Total images: {args.num_batches * args.images_per_batch}")
    print()
    print("Configurations to test:")
    for dp, tp in configs:
        total_gpus = dp + tp
        print(f"  Vision DP={dp}, LLM TP={tp} ({total_gpus} GPUs)")
    print()
    print("Communication:")
    print("  - NCCL all_gather for vision embeddings")
    print("  - NCCL P2P send/recv for vision-to-LLM transfer")
    print("="*100)
    
    all_results = []
    base_port = 29500
    
    for idx, (vision_dp, llm_tp) in enumerate(configs):
        results = benchmark_config(
            model_name=args.model_name,
            vision_dp=vision_dp,
            llm_tp=llm_tp,
            num_batches=args.num_batches,
            images_per_batch=args.images_per_batch,
            warmup_batches=args.warmup_batches,
            master_port=base_port + idx * 100,
        )
        print_results(results)
        all_results.append(results)
        
        # Wait between configs (longer to ensure NCCL cleanup)
        if idx < len(configs) - 1:
            logger.info("Waiting for NCCL cleanup between configs...")
            time.sleep(10)
    
    # Print comparison table if multiple configs
    if len(all_results) > 1:
        print_comparison_table(all_results)
    
    print("="*100)
    print("Benchmark complete!")
    print("="*100)


if __name__ == "__main__":
    main()
