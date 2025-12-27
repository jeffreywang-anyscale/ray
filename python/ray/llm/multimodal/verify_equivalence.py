"""
Verification Script: Verify output equivalence across DP/TP configurations.

This script verifies that different parallelism configurations produce
identical outputs to the baseline (single GPU, no DP/TP).

Test cases:
1. Baseline: Single GPU (DP=1, TP=1)
2. Non-colocated: Vision DP + LLM TP on separate GPUs
3. Colocated: Vision DP + LLM TP on same GPUs

All configurations should produce equivalent vision embeddings and LLM outputs.
"""

import argparse
import logging
import sys
import time
from typing import Tuple

import ray
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
    """Create a dummy batch of image data."""
    h_patches = image_height // patch_size
    w_patches = image_width // patch_size
    t_patches = 1
    
    patches_per_image = t_patches * h_patches * w_patches
    total_patches = num_images * patches_per_image
    patch_dim = temporal_patch_size * in_channels * patch_size * patch_size
    
    torch.manual_seed(42)
    pixel_values = torch.randn(total_patches, patch_dim, dtype=torch.float32)
    grid_thw = torch.tensor(
        [[t_patches, h_patches, w_patches] for _ in range(num_images)],
        dtype=torch.long
    )
    
    return pixel_values, grid_thw


def compare_tensors(
    t1: torch.Tensor,
    t2: torch.Tensor,
    name: str,
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> bool:
    """Compare two tensors and report differences."""
    t1 = t1.cpu().float()
    t2 = t2.cpu().float()
    
    # Handle batch dimension
    if t1.dim() != t2.dim():
        if t1.dim() == 3 and t2.dim() == 2:
            t1 = t1.squeeze(0)
        elif t2.dim() == 3 and t1.dim() == 2:
            t2 = t2.squeeze(0)
    
    if t1.shape != t2.shape:
        logger.error(f"{name}: Shape mismatch - {t1.shape} vs {t2.shape}")
        return False
    
    try:
        is_close = torch.allclose(t1, t2, rtol=rtol, atol=atol)
    except RuntimeError as e:
        logger.error(f"{name}: Comparison failed - {e}")
        return False
    
    if is_close:
        logger.info(f"✓ {name}: Tensors are equivalent (rtol={rtol}, atol={atol})")
    else:
        abs_diff = torch.abs(t1 - t2)
        max_diff = abs_diff.max().item()
        mean_diff = abs_diff.mean().item()
        logger.error(f"✗ {name}: Tensors differ!")
        logger.error(f"  Max absolute diff: {max_diff:.6e}")
        logger.error(f"  Mean absolute diff: {mean_diff:.6e}")
    
    return is_close


def run_baseline(
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    model_name: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run baseline: Single GPU with DP=1, TP=1.
    This is the reference that all other configurations should match.
    """
    from qwen3_vl_vision_actor import create_vision_actor_class
    from qwen3_vl_llm_actor import create_llm_actor_class
    
    logger.info("BASELINE: Single GPU, DP=1, TP=1")
    
    # Create single vision actor
    VisionActorClass = create_vision_actor_class(num_gpus=1, num_cpus=4)
    vision_actor = VisionActorClass.remote(
        model_name, dp_rank=0, dp_size=1, global_rank=0
    )
    
    # Create single LLM actor
    LLMActorClass = create_llm_actor_class(num_gpus=1, num_cpus=4)
    llm_actor = LLMActorClass.remote(
        model_name, tp_rank=0, tp_size=1, global_rank=1
    )
    
    # Build models (no distributed needed for single GPU)
    ray.get(vision_actor.build_model.remote())
    ray.get(llm_actor.build_model.remote())
    logger.info("BASELINE: Models built")
    
    # Run vision forward
    vision_embeddings = ray.get(vision_actor.forward.remote(pixel_values, grid_thw))
    logger.info(f"BASELINE: Vision embeddings shape {vision_embeddings.shape}")
    
    # Run LLM forward (pass embeddings directly, no distributed transfer)
    llm_output = ray.get(llm_actor.forward.remote(vision_embeddings))
    logger.info(f"BASELINE: LLM output shape {llm_output.shape}")
    
    # Cleanup
    ray.kill(vision_actor)
    ray.kill(llm_actor)
    
    return vision_embeddings, llm_output


def run_non_colocated(
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    model_name: str,
    vision_dp: int,
    llm_tp: int,
    master_port: int = 29510,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run non-colocated: Vision DP on GPUs 0..dp-1, LLM TP on GPUs dp..dp+tp-1.
    """
    from qwen3_vl_vision_actor import create_vision_actor_class
    from qwen3_vl_llm_actor import create_llm_actor_class
    
    world_size = vision_dp + llm_tp
    
    logger.info(f"NON-COLOCATED: Vision DP={vision_dp}, LLM TP={llm_tp}")
    logger.info(f"  Vision GPUs: 0 to {vision_dp - 1}")
    logger.info(f"  LLM GPUs: {vision_dp} to {world_size - 1}")
    
    # Create vision actors
    VisionActorClass = create_vision_actor_class(num_gpus=1, num_cpus=4)
    vision_actors = [
        VisionActorClass.remote(model_name, dp_rank=i, dp_size=vision_dp, global_rank=i)
        for i in range(vision_dp)
    ]
    
    # Create LLM actors
    LLMActorClass = create_llm_actor_class(num_gpus=1, num_cpus=4)
    llm_actors = [
        LLMActorClass.remote(model_name, tp_rank=i, tp_size=llm_tp, global_rank=vision_dp + i)
        for i in range(llm_tp)
    ]
    
    # Initialize distributed
    master_addr = ray.get(vision_actors[0].get_ip_address.remote())
    
    vision_init = [
        actor.init_distributed.remote(master_addr, master_port, world_size, llm_tp_size=llm_tp)
        for actor in vision_actors
    ]
    llm_init = [
        actor.init_distributed.remote(master_addr, master_port, world_size, vision_dp_size=vision_dp)
        for actor in llm_actors
    ]
    ray.get(vision_init + llm_init)
    logger.info(f"NON-COLOCATED: Distributed initialized (world_size={world_size})")
    
    # Build models
    ray.get([actor.build_model.remote() for actor in vision_actors + llm_actors])
    logger.info(f"NON-COLOCATED: Models built")
    
    # Split batch for DP
    num_images = grid_thw.shape[0]
    images_per_actor = num_images // vision_dp
    extra = num_images % vision_dp
    patches_per_image = grid_thw[0, 0].item() * grid_thw[0, 1].item() * grid_thw[0, 2].item()
    
    # Vision forward
    vision_refs = []
    pixel_offset = 0
    image_offset = 0
    for i in range(vision_dp):
        num_imgs = images_per_actor + (1 if i < extra else 0)
        num_patches = num_imgs * patches_per_image
        chunk_pixels = pixel_values[pixel_offset:pixel_offset + num_patches]
        chunk_grid = grid_thw[image_offset:image_offset + num_imgs]
        vision_refs.append(vision_actors[i].forward.remote(chunk_pixels, chunk_grid))
        pixel_offset += num_patches
        image_offset += num_imgs
    
    local_outputs = ray.get(vision_refs)
    
    # All-gather
    if vision_dp > 1:
        gather_refs = [
            vision_actors[i].all_gather_embeddings.remote(local_outputs[i])
            for i in range(vision_dp)
        ]
        full_outputs = ray.get(gather_refs)
        vision_embeddings = full_outputs[0]
    else:
        vision_embeddings = local_outputs[0]
    logger.info(f"NON-COLOCATED: Vision embeddings gathered, shape {vision_embeddings.shape}")
    
    # P2P transfer to LLM actors
    embedding_shape = (vision_embeddings.shape[0], vision_embeddings.shape[1])
    send_refs = []
    recv_refs = []
    for i in range(llm_tp):
        dst_rank = vision_dp + i
        send_refs.append(vision_actors[0].send_embeddings_p2p.remote(vision_embeddings, dst_rank))
        recv_refs.append(llm_actors[i].recv_embeddings_p2p.remote(embedding_shape, src_rank=0))
    
    ray.get(send_refs)
    received = ray.get(recv_refs)
    logger.info(f"NON-COLOCATED: Embeddings transferred via P2P")
    
    # LLM forward
    llm_refs = [llm_actors[i].forward.remote(received[i]) for i in range(llm_tp)]
    llm_outputs = ray.get(llm_refs)
    llm_output = llm_outputs[0]
    logger.info(f"NON-COLOCATED: LLM output shape {llm_output.shape}")
    
    # Cleanup
    cleanup = [actor.cleanup_distributed.remote() for actor in vision_actors + llm_actors]
    ray.get(cleanup)
    for actor in vision_actors + llm_actors:
        ray.kill(actor)
    time.sleep(1)
    
    return vision_embeddings, llm_output


def run_colocated(
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    model_name: str,
    dp_tp_size: int,
    master_port: int = 29520,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run colocated: Vision DP and LLM TP on same GPUs.
    LLM actors don't use NCCL - they receive embeddings via Ray.
    """
    from qwen3_vl_vision_actor import create_vision_actor_class
    from qwen3_vl_llm_actor import create_llm_actor_class
    
    logger.info(f"COLOCATED: DP=TP={dp_tp_size}")
    logger.info(f"  GPUs: 0 to {dp_tp_size - 1} (shared)")
    
    # Create vision actors
    VisionActorClass = create_vision_actor_class(num_gpus=1, num_cpus=4)
    vision_actors = [
        VisionActorClass.remote(model_name, dp_rank=i, dp_size=dp_tp_size, global_rank=i)
        for i in range(dp_tp_size)
    ]
    
    # Create LLM actors (share GPU, no NCCL)
    LLMActorClass = create_llm_actor_class(num_gpus=0.5, num_cpus=2)
    llm_actors = [
        LLMActorClass.remote(model_name, tp_rank=i, tp_size=1, global_rank=i)
        for i in range(dp_tp_size)
    ]
    
    # Only vision actors need distributed (for DP all_gather)
    if dp_tp_size > 1:
        master_addr = ray.get(vision_actors[0].get_ip_address.remote())
        init_refs = [
            actor.init_distributed.remote(master_addr, master_port, dp_tp_size, llm_tp_size=0)
            for actor in vision_actors
        ]
        ray.get(init_refs)
        logger.info(f"COLOCATED: Vision distributed initialized (world_size={dp_tp_size})")
    
    # Build models
    ray.get([actor.build_model.remote() for actor in vision_actors + llm_actors])
    logger.info(f"COLOCATED: Models built")
    
    # Split batch for DP
    num_images = grid_thw.shape[0]
    images_per_actor = num_images // dp_tp_size
    extra = num_images % dp_tp_size
    patches_per_image = grid_thw[0, 0].item() * grid_thw[0, 1].item() * grid_thw[0, 2].item()
    
    # Vision forward
    vision_refs = []
    pixel_offset = 0
    image_offset = 0
    for i in range(dp_tp_size):
        num_imgs = images_per_actor + (1 if i < extra else 0)
        num_patches = num_imgs * patches_per_image
        chunk_pixels = pixel_values[pixel_offset:pixel_offset + num_patches]
        chunk_grid = grid_thw[image_offset:image_offset + num_imgs]
        vision_refs.append(vision_actors[i].forward.remote(chunk_pixels, chunk_grid))
        pixel_offset += num_patches
        image_offset += num_imgs
    
    local_outputs = ray.get(vision_refs)
    
    # All-gather
    if dp_tp_size > 1:
        gather_refs = [
            vision_actors[i].all_gather_embeddings.remote(local_outputs[i])
            for i in range(dp_tp_size)
        ]
        full_outputs = ray.get(gather_refs)
    else:
        full_outputs = local_outputs
    
    vision_embeddings = full_outputs[0]
    logger.info(f"COLOCATED: Vision embeddings gathered, shape {vision_embeddings.shape}")
    
    # LLM forward (each gets embeddings from colocated vision)
    llm_refs = [llm_actors[i].forward.remote(full_outputs[i]) for i in range(dp_tp_size)]
    llm_outputs = ray.get(llm_refs)
    llm_output = llm_outputs[0]
    logger.info(f"COLOCATED: LLM output shape {llm_output.shape}")
    
    # Cleanup - only vision actors have distributed
    if dp_tp_size > 1:
        cleanup = [actor.cleanup_distributed.remote() for actor in vision_actors]
        ray.get(cleanup)
    for actor in vision_actors + llm_actors:
        ray.kill(actor)
    time.sleep(1)
    
    return vision_embeddings, llm_output


def verify_all(
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    model_name: str,
    rtol: float,
    atol: float,
) -> bool:
    """Run all verification tests."""
    
    # Run baseline
    print("\n" + "="*70)
    print("CASE 1: BASELINE (DP=1, TP=1)")
    print("="*70)
    baseline_vision, baseline_llm = run_baseline(pixel_values, grid_thw, model_name)
    
    results = {}
    
    # Test non-colocated DP=2, TP=1
    print("\n" + "="*70)
    print("CASE 2: NON-COLOCATED (DP=2, TP=1)")
    print("="*70)
    try:
        nc_vision_21, nc_llm_21 = run_non_colocated(
            pixel_values, grid_thw, model_name,
            vision_dp=2, llm_tp=1, master_port=29511
        )
        v_match = compare_tensors(baseline_vision, nc_vision_21, "Vision (DP=2,TP=1)", rtol, atol)
        l_match = compare_tensors(baseline_llm, nc_llm_21, "LLM (DP=2,TP=1)", rtol, atol)
        results["non-colocated DP=2,TP=1"] = v_match and l_match
    except Exception as e:
        logger.error(f"CASE 2 failed: {e}")
        results["non-colocated DP=2,TP=1"] = False
    
    # Test non-colocated DP=1, TP=2
    print("\n" + "="*70)
    print("CASE 3: NON-COLOCATED (DP=1, TP=2)")
    print("="*70)
    try:
        nc_vision_12, nc_llm_12 = run_non_colocated(
            pixel_values, grid_thw, model_name,
            vision_dp=1, llm_tp=2, master_port=29512
        )
        v_match = compare_tensors(baseline_vision, nc_vision_12, "Vision (DP=1,TP=2)", rtol, atol)
        l_match = compare_tensors(baseline_llm, nc_llm_12, "LLM (DP=1,TP=2)", rtol, atol)
        results["non-colocated DP=1,TP=2"] = v_match and l_match
    except Exception as e:
        logger.error(f"CASE 3 failed: {e}")
        results["non-colocated DP=1,TP=2"] = False
    
    # Test non-colocated DP=2, TP=2
    print("\n" + "="*70)
    print("CASE 4: NON-COLOCATED (DP=2, TP=2)")
    print("="*70)
    try:
        nc_vision_22, nc_llm_22 = run_non_colocated(
            pixel_values, grid_thw, model_name,
            vision_dp=2, llm_tp=2, master_port=29513
        )
        v_match = compare_tensors(baseline_vision, nc_vision_22, "Vision (DP=2,TP=2)", rtol, atol)
        l_match = compare_tensors(baseline_llm, nc_llm_22, "LLM (DP=2,TP=2)", rtol, atol)
        results["non-colocated DP=2,TP=2"] = v_match and l_match
    except Exception as e:
        logger.error(f"CASE 4 failed: {e}")
        results["non-colocated DP=2,TP=2"] = False
    
    # Test colocated DP=TP=2
    print("\n" + "="*70)
    print("CASE 5: COLOCATED (DP=TP=2)")
    print("="*70)
    try:
        c_vision_2, c_llm_2 = run_colocated(
            pixel_values, grid_thw, model_name,
            dp_tp_size=2, master_port=29514
        )
        v_match = compare_tensors(baseline_vision, c_vision_2, "Vision (colocated)", rtol, atol)
        l_match = compare_tensors(baseline_llm, c_llm_2, "LLM (colocated)", rtol, atol)
        results["colocated DP=TP=2"] = v_match and l_match
    except Exception as e:
        logger.error(f"CASE 5 failed: {e}")
        results["colocated DP=TP=2"] = False
    
    # Summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    all_passed = True
    for name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")
        all_passed = all_passed and passed
    
    print("="*70)
    if all_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("="*70)
    
    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Verify output equivalence across DP/TP configs")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--num_images", type=int, default=4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--vision_dp", type=int, default=None, help="Custom vision DP")
    parser.add_argument("--llm_tp", type=int, default=None, help="Custom LLM TP")
    parser.add_argument("--colocated", action="store_true", help="Test colocated mode")
    
    args = parser.parse_args()
    
    # Ensure divisible by 2 for DP tests
    if args.num_images % 2 != 0:
        args.num_images = (args.num_images // 2 + 1) * 2
        logger.warning(f"Adjusted num_images to {args.num_images}")
    
    print("="*70)
    print("Verification: Output Equivalence across DP/TP Configurations")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Number of images: {args.num_images}")
    print(f"Tolerance: rtol={args.rtol}, atol={args.atol}")
    print("="*70)
    
    if not ray.is_initialized():
        ray.init()
    
    # Create dummy batch
    pixel_values, grid_thw = create_dummy_batch(args.num_images)
    logger.info(f"Batch created: pixels={pixel_values.shape}, grid_thw={grid_thw.shape}")
    
    # Custom test or full suite
    if args.vision_dp is not None and args.llm_tp is not None:
        print("\n" + "="*70)
        print(f"CUSTOM TEST: DP={args.vision_dp}, TP={args.llm_tp}, colocated={args.colocated}")
        print("="*70)
        
        baseline_vision, baseline_llm = run_baseline(pixel_values, grid_thw, args.model)
        
        if args.colocated:
            if args.vision_dp != args.llm_tp:
                print("ERROR: Colocated requires DP == TP")
                return 1
            test_vision, test_llm = run_colocated(
                pixel_values, grid_thw, args.model,
                dp_tp_size=args.vision_dp, master_port=29515
            )
        else:
            test_vision, test_llm = run_non_colocated(
                pixel_values, grid_thw, args.model,
                vision_dp=args.vision_dp, llm_tp=args.llm_tp, master_port=29515
            )
        
        v_match = compare_tensors(baseline_vision, test_vision, "Vision", args.rtol, args.atol)
        l_match = compare_tensors(baseline_llm, test_llm, "LLM", args.rtol, args.atol)
        
        passed = v_match and l_match
        print("\n" + "="*70)
        print(f"RESULT: {'✓ PASSED' if passed else '✗ FAILED'}")
        print("="*70)
        return 0 if passed else 1
    
    # Full test suite
    passed = verify_all(pixel_values, grid_thw, args.model, args.rtol, args.atol)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
