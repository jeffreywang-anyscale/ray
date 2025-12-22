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
    image_height: int = 28,
    image_width: int = 28,
    patch_size: int = 14,
    temporal_patch_size: int = 2,
    in_channels: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a dummy batch of images for benchmarking.
    Matches verify_equivalence.py format.
    """
    h_patches = image_height // patch_size
    w_patches = image_width // patch_size
    t_patches = 1  # Single frame for images
    
    patches_per_image = t_patches * h_patches * w_patches
    total_patches = num_images * patches_per_image
    
    # Patch dimension: temporal_patch_size * in_channels * patch_size * patch_size
    patch_dim = temporal_patch_size * in_channels * patch_size * patch_size
    
    # Create pixel values [total_patches, patch_dim]
    torch.manual_seed(42)
    pixel_values = torch.randn(total_patches, patch_dim, dtype=torch.float32)
    
    # Create grid_thw [num_images, 3]
    grid_thw = torch.tensor([[t_patches, h_patches, w_patches]] * num_images, dtype=torch.long)
    
    return pixel_values, grid_thw


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
    pixel_values, grid_thw = create_dummy_batch(images_per_batch)
    
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
    
    results['throughput'] = {
        'images_per_sec': total_images / total_time,
        'batches_per_sec': num_batches / total_time,
        'vision_images_per_sec': total_images / sum(vision_times),
        'llm_batches_per_sec': num_batches / sum(llm_times),
    }
    
    return results


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
    print("  Throughput:")
    t = results['throughput']
    print(f"    End-to-end: {t['images_per_sec']:.2f} images/sec, {t['batches_per_sec']:.2f} batches/sec")
    print(f"    Vision encoder: {t['vision_images_per_sec']:.2f} images/sec")
    print(f"    LLM decoder: {t['llm_batches_per_sec']:.2f} batches/sec")
    print()


def print_comparison_table(all_results: List[Dict[str, Any]], title: str = "COMPARISON TABLE"):
    """Print comparison table for multiple configurations."""
    print(f"\n{'='*100}")
    print(title)
    print(f"{'='*100}")
    
    # Header
    print(f"\n  {'Config':<15} {'Batch':>6} {'GPUs':>5} {'Vision(s)':>10} {'LLM(s)':>10} {'Total(s)':>10} {'img/sec':>10} {'Vision img/s':>12}")
    print(f"  {'-'*90}")
    
    for r in all_results:
        config = r['config']
        batch = r['images_per_batch']
        gpus = r['total_gpus']
        vision = r['vision']['mean']
        llm = r['llm']['mean']
        total = r['total']['mean']
        throughput = r['throughput']['images_per_sec']
        vision_tp = r['throughput']['vision_images_per_sec']
        print(f"  {config:<15} {batch:>6} {gpus:>5} {vision:>10.4f} {llm:>10.4f} {total:>10.4f} {throughput:>10.2f} {vision_tp:>12.2f}")
    
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
