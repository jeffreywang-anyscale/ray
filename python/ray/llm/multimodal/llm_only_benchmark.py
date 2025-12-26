#!/usr/bin/env python3
"""
LLM decoder only benchmark.
Measures the maximum req/s a single LLM replica can yield.
"""

import argparse
import logging
import time

import ray
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_dummy_embeddings(
    num_tokens: int,
    hidden_size: int = 3584,  # Qwen3-VL-7B hidden size
) -> torch.Tensor:
    """
    Create dummy vision embeddings to feed to LLM.
    Simulates output from vision encoder.
    
    Args:
        num_tokens: Number of visual tokens (e.g., 256 for 448x448 image)
        hidden_size: Model hidden dimension
    
    Returns:
        embeddings: [1, num_tokens, hidden_size] - batch size 1
    """
    torch.manual_seed(42)
    # Shape: [batch=1, seq_len=num_tokens, hidden=3584]
    embeddings = torch.randn(1, num_tokens, hidden_size, dtype=torch.bfloat16)
    return embeddings


def benchmark_llm_only(
    model_name: str,
    tokens_per_request: int,
    num_batches: int,
    warmup_batches: int,
):
    """Benchmark LLM decoder only - no vision encoder, no distributed comm."""
    
    from qwen3_vl_llm_actor import create_llm_actor_class
    
    # Create single LLM actor (TP=1) - same as disaggregated benchmark
    LLMActorCls = create_llm_actor_class(num_gpus=1, num_cpus=4)
    llm_actor = LLMActorCls.remote(
        model_name,  # positional arg like disaggregated benchmark
        tp_rank=0,
        tp_size=1,
        global_rank=0,  # No vision actors, so global_rank = tp_rank
    )
    
    # Build model
    logger.info("Building LLM decoder...")
    ray.get(llm_actor.build_model.remote())
    logger.info("LLM decoder ready")
    
    # Get hidden size from the model
    hidden_size = ray.get(llm_actor.get_hidden_size.remote())
    logger.info(f"Model hidden size: {hidden_size}")
    
    # Create dummy embeddings (simulating vision encoder output)
    # Shape: [1, tokens_per_request, hidden_size]
    embeddings = create_dummy_embeddings(tokens_per_request, hidden_size=hidden_size)
    
    logger.info(f"Input: {tokens_per_request} tokens per request")
    logger.info(f"Embeddings shape: {embeddings.shape}")
    
    # Warmup
    logger.info(f"Warming up ({warmup_batches} batches)...")
    for _ in range(warmup_batches):
        ray.get(llm_actor.forward.remote(embeddings))
    
    # Benchmark
    logger.info(f"Benchmarking ({num_batches} batches)...")
    times = []
    for i in range(num_batches):
        start = time.perf_counter()
        ray.get(llm_actor.forward.remote(embeddings))
        end = time.perf_counter()
        times.append(end - start)
        
        if (i + 1) % 50 == 0:
            logger.info(f"  Completed {i + 1}/{num_batches} batches")
    
    # Cleanup
    ray.kill(llm_actor)
    
    # Calculate stats
    total_time = sum(times)
    avg_time = total_time / num_batches
    min_time = min(times)
    max_time = max(times)
    
    req_per_sec = num_batches / total_time
    tokens_per_sec = (num_batches * tokens_per_request) / total_time
    
    # Print results
    print("\n" + "=" * 60)
    print("LLM DECODER BENCHMARK (Single Replica, TP=1)")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Tokens per request:   {tokens_per_request}")
    print(f"  Num batches:          {num_batches}")
    
    print(f"\nLatency (seconds/request):")
    print(f"  Mean:    {avg_time:.4f}")
    print(f"  Min:     {min_time:.4f}")
    print(f"  Max:     {max_time:.4f}")
    
    print(f"\nThroughput:")
    print(f"  Requests/sec:   {req_per_sec:.2f}")
    print(f"  Tokens/sec:     {tokens_per_sec:.2f}")
    print("=" * 60)
    
    return {
        'req_per_sec': req_per_sec,
        'tokens_per_sec': tokens_per_sec,
        'latency_mean': avg_time,
        'latency_min': min_time,
        'latency_max': max_time,
    }


def main():
    parser = argparse.ArgumentParser(description="LLM decoder only benchmark")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--tokens_per_request", type=int, default=256, 
                        help="Tokens per request (256 = 1 image @ 448x448)")
    parser.add_argument("--num_batches", type=int, default=100, help="Number of batches to run")
    parser.add_argument("--warmup_batches", type=int, default=10, help="Warmup batches")
    args = parser.parse_args()
    
    if not ray.is_initialized():
        ray.init()
    
    try:
        benchmark_llm_only(
            model_name=args.model_name,
            tokens_per_request=args.tokens_per_request,
            num_batches=args.num_batches,
            warmup_batches=args.warmup_batches,
        )
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()

