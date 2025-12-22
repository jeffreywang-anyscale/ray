"""
Verification Script: Compare Disaggregated vs Combined Approaches

This script verifies that the disaggregated approach (encoder and decoder on DIFFERENT actors)
produces identical outputs to the combined approach (encoder and decoder on the SAME actor).

All cases use NCCL all_gather for vision embeddings.

Cases tested:
4. Configurable DP/TP: Compare baseline (single GPU) with various DP/TP configurations
5. NCCL Combined: Vision DP + NCCL all_gather + LLM TP on SAME GPUs (0,1)
6. Combined vs Disaggregated: Vision+LLM on GPU 0,1 vs Vision GPU 0,1 + LLM GPU 2,3

GPU Placement:
- Combined: Vision and LLM on SAME GPUs (e.g., GPU 0,1 for both)
- Disaggregated: Vision on GPUs 0,1, LLM on GPUs 2,3 (separate GPUs)

Constraint: Total 4 GPUs available.
"""

import argparse
import logging
import sys
from typing import List, Tuple

import ray
import torch

# Setup logging
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
    """
    Create a dummy batch of image data.
    
    Returns:
        Tuple of (pixel_values, grid_thw)
    """
    # Calculate patches per image
    h_patches = image_height // patch_size
    w_patches = image_width // patch_size
    t_patches = 1  # Single frame for images
    
    patches_per_image = t_patches * h_patches * w_patches
    total_patches = num_images * patches_per_image
    
    # Patch dimension: temporal_patch_size * in_channels * patch_size * patch_size
    patch_dim = temporal_patch_size * in_channels * patch_size * patch_size
    
    # Create pixel values [total_patches, patch_dim]
    # Use deterministic values for reproducibility
    torch.manual_seed(42)
    pixel_values = torch.randn(total_patches, patch_dim, dtype=torch.float32)
    
    # Create grid_thw [num_images, 3] - (temporal, height, width) patches per image
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
    """
    Compare two tensors and report differences.
    
    Returns:
        True if tensors are close enough, False otherwise
    """
    # Ensure same device and dtype for comparison
    t1 = t1.cpu().float()
    t2 = t2.cpu().float()
    
    # Handle batch dimension differences
    if t1.dim() != t2.dim():
        if t1.dim() == 3 and t2.dim() == 2:
            t1 = t1.squeeze(0)
        elif t2.dim() == 3 and t1.dim() == 2:
            t2 = t2.squeeze(0)
    
    # Check shapes
    if t1.shape != t2.shape:
        logger.error(f"{name}: Shape mismatch - {t1.shape} vs {t2.shape}")
        return False
    
    # Check values
    try:
        is_close = torch.allclose(t1, t2, rtol=rtol, atol=atol)
    except RuntimeError as e:
        logger.error(f"{name}: Comparison failed - {e}")
        return False
    
    if is_close:
        logger.info(f"✓ {name}: Tensors are equivalent (rtol={rtol}, atol={atol})")
    else:
        # Compute differences
        abs_diff = torch.abs(t1 - t2)
        max_diff = abs_diff.max().item()
        mean_diff = abs_diff.mean().item()
        
        rel_diff = abs_diff / (torch.abs(t2) + 1e-10)
        max_rel_diff = rel_diff.max().item()
        mean_rel_diff = rel_diff.mean().item()
        
        logger.error(f"✗ {name}: Tensors differ!")
        logger.error(f"  Max absolute diff: {max_diff:.6e}")
        logger.error(f"  Mean absolute diff: {mean_diff:.6e}")
        logger.error(f"  Max relative diff: {max_rel_diff:.6e}")
        logger.error(f"  Mean relative diff: {mean_rel_diff:.6e}")
    
    return is_close


def run_baseline_single_gpu(
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    model_name: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run baseline: Single GPU with no DP and no TP.
    
    This is the reference implementation that all other configurations
    should match exactly.
    
    Total GPUs: 1 (GPU 0)
    
    Returns:
        Tuple of (vision_embeddings, llm_output)
    """
    from qwen_vl_combined_actor import create_combined_actor_class
    
    logger.info("BASELINE: Single GPU, no DP, no TP")
    
    # Create single actor on GPU 0
    CombinedActorClass = create_combined_actor_class(num_gpus=1, num_cpus=4)
    actor = CombinedActorClass.remote(model_name, rank=0, tp_size=1)
    
    # Build model
    ray.get(actor.build_model.remote())
    logger.info("BASELINE: Model built on single GPU")
    
    # Run full forward pass
    vision_embeddings, llm_output = ray.get(
        actor.forward_full.remote(pixel_values, grid_thw)
    )
    logger.info(f"BASELINE: Vision embeddings shape {vision_embeddings.shape}")
    logger.info(f"BASELINE: LLM output shape {llm_output.shape}")
    
    # Cleanup
    ray.kill(actor)
    
    return vision_embeddings, llm_output


def run_nccl_disaggregated(
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    model_name: str,
    vision_dp: int,
    llm_tp: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run disaggregated approach with NCCL all_gather for vision embeddings
    and NCCL P2P for vision-to-LLM transfer.
    
    Vision actors on GPUs: 0, 1, ..., (vision_dp - 1)
    LLM actors on GPUs: vision_dp, vision_dp + 1, ..., (vision_dp + llm_tp - 1)
    
    Uses:
    - NCCL all_gather among vision actors
    - NCCL P2P send/recv from vision actor 0 to all LLM actors
    
    Total GPUs: vision_dp + llm_tp
    
    Args:
        pixel_values: Input pixel values
        grid_thw: Grid dimensions
        model_name: Model name or path
        vision_dp: Number of vision DP actors
        llm_tp: Tensor parallel size for LLM
    
    Returns:
        Tuple of (vision_embeddings, llm_output)
    """
    from qwen3_vl_vision_actor import create_vision_actor_class
    from qwen3_vl_llm_actor import create_llm_actor_class
    
    total_world_size = vision_dp + llm_tp
    
    logger.info(
        f"NCCL DISAGGREGATED: "
        f"{vision_dp} vision actors (DP) + {llm_tp} LLM actors (TP)"
    )
    logger.info(f"  Vision GPUs: 0 to {vision_dp - 1}")
    logger.info(f"  LLM GPUs: {vision_dp} to {vision_dp + llm_tp - 1}")
    logger.info(f"  Using NCCL all_gather for vision embeddings")
    logger.info(f"  Using NCCL P2P for vision-to-LLM transfer")
    
    # Create vision actors with DP size
    VisionActorClass = create_vision_actor_class(num_gpus=1, num_cpus=4)
    vision_actors = [
        VisionActorClass.remote(model_name, rank=i, dp_size=vision_dp)
        for i in range(vision_dp)
    ]
    
    # Create LLM actors with global ranks
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
    master_port = 29506
    
    # Initialize distributed for ALL actors with world_size = vision_dp + llm_tp
    # This allows P2P communication between vision and LLM actors
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
    logger.info(f"NCCL DISAGGREGATED: All actors distributed initialized (world_size={total_world_size})")
    
    # Build models
    vision_build_refs = [actor.build_model.remote() for actor in vision_actors]
    llm_build_refs = [actor.build_model.remote() for actor in llm_actors]
    ray.get(vision_build_refs + llm_build_refs)
    logger.info(f"NCCL DISAGGREGATED: All models built")
    
    # Split batch across vision actors (DP)
    num_images = grid_thw.shape[0]
    images_per_actor = num_images // vision_dp
    extra = num_images % vision_dp
    patches_per_image = grid_thw[0, 0].item() * grid_thw[0, 1].item() * grid_thw[0, 2].item()
    
    # Step 1: Run vision forward on each actor
    vision_refs = []
    pixel_offset = 0
    image_offset = 0
    
    for i in range(vision_dp):
        num_imgs = images_per_actor + (1 if i < extra else 0)
        num_patches = num_imgs * patches_per_image
        
        chunk_pixels = pixel_values[pixel_offset:pixel_offset + num_patches]
        chunk_grid = grid_thw[image_offset:image_offset + num_imgs]
        
        ref = vision_actors[i].forward.remote(chunk_pixels, chunk_grid)
        vision_refs.append(ref)
        
        pixel_offset += num_patches
        image_offset += num_imgs
    
    # Get local vision embeddings
    local_vision_outputs = ray.get(vision_refs)
    
    # Step 2: NCCL all_gather among vision actors
    if vision_dp > 1:
        all_gather_refs = [
            vision_actors[i].all_gather_embeddings.remote(local_vision_outputs[i])
            for i in range(vision_dp)
        ]
        full_vision_outputs = ray.get(all_gather_refs)
        vision_embeddings = full_vision_outputs[0]
    else:
        vision_embeddings = local_vision_outputs[0]
    logger.info(f"NCCL DISAGGREGATED: Vision embeddings gathered via NCCL, shape {vision_embeddings.shape}")
    
    # Step 3: Vision actor 0 sends embeddings to ALL LLM actors via NCCL P2P
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
    logger.info(f"NCCL DISAGGREGATED: Embeddings transferred via NCCL P2P")
    
    # Step 4: Run LLM forward on all TP ranks
    llm_refs = [
        actor.forward.remote(received_embeddings[i])
        for i, actor in enumerate(llm_actors)
    ]
    llm_outputs = ray.get(llm_refs)
    
    # All TP ranks should have identical output after all-reduce
    llm_output = llm_outputs[0]
    logger.info(f"NCCL DISAGGREGATED: LLM output shape {llm_output.shape}")
    
    # Cleanup distributed before killing actors
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
    
    import time
    time.sleep(1)  # Allow NCCL resources to be released
    
    return vision_embeddings, llm_output


def run_nccl_combined(
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    model_name: str,
    num_actors: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run combined approach with NCCL-based all_gather for vision embeddings.
    
    Each actor has both vision encoder and LLM on the SAME GPU.
    Vision encoders do DP (each processes portion of batch).
    NCCL all_gather combines embeddings so each actor has full batch.
    LLM does TP (all-reduce on outputs).
    
    GPUs used: 0, 1, ..., (num_actors - 1)
    Both vision and LLM on same GPUs.
    
    Args:
        pixel_values: Input pixel values
        grid_thw: Grid dimensions
        model_name: Model name or path
        num_actors: Number of combined actors (DP for vision, TP for LLM)
    
    Returns:
        Tuple of (vision_embeddings, llm_output)
    """
    from qwen_vl_combined_actor import create_combined_actor_class
    
    logger.info(
        f"NCCL COMBINED: {num_actors} actors "
        f"(Vision DP + NCCL all_gather + LLM TP on same GPUs)"
    )
    logger.info(f"  GPUs: 0 to {num_actors - 1} (shared by vision and LLM)")
    logger.info(f"  Using NCCL all_gather for vision embeddings")
    
    # Create combined actors
    CombinedActorClass = create_combined_actor_class(num_gpus=1, num_cpus=4)
    actors = [
        CombinedActorClass.remote(model_name, rank=i, tp_size=num_actors)
        for i in range(num_actors)
    ]
    
    # Initialize distributed for NCCL all_gather and TP
    if num_actors > 1:
        master_addr = ray.get(actors[0].get_ip_address.remote())
        master_port = 29504
        
        init_refs = [
            actor.init_distributed.remote(master_addr, master_port)
            for actor in actors
        ]
        ray.get(init_refs)
        logger.info(f"NCCL COMBINED: Distributed initialized (world_size={num_actors})")
    
    # Build models
    build_refs = [actor.build_model.remote() for actor in actors]
    ray.get(build_refs)
    logger.info(f"NCCL COMBINED: Models built")
    
    # Split batch across actors for vision (DP)
    num_images = grid_thw.shape[0]
    images_per_actor = num_images // num_actors
    extra = num_images % num_actors
    patches_per_image = grid_thw[0, 0].item() * grid_thw[0, 1].item() * grid_thw[0, 2].item()
    
    # Step 1: Run vision forward on each actor (DP - each processes different images)
    vision_refs = []
    pixel_offset = 0
    image_offset = 0
    
    for i in range(num_actors):
        num_imgs = images_per_actor + (1 if i < extra else 0)
        num_patches = num_imgs * patches_per_image
        
        chunk_pixels = pixel_values[pixel_offset:pixel_offset + num_patches]
        chunk_grid = grid_thw[image_offset:image_offset + num_imgs]
        
        ref = actors[i].forward_vision.remote(chunk_pixels, chunk_grid)
        vision_refs.append(ref)
        
        pixel_offset += num_patches
        image_offset += num_imgs
    
    # Get local vision embeddings
    local_vision_outputs = ray.get(vision_refs)
    
    # Step 2: NCCL all_gather - each actor gets FULL vision embeddings
    all_gather_refs = [
        actors[i].all_gather_vision_embeddings.remote(local_vision_outputs[i])
        for i in range(num_actors)
    ]
    full_vision_outputs = ray.get(all_gather_refs)
    
    # All actors now have the same full embeddings
    vision_embeddings = full_vision_outputs[0]
    logger.info(f"NCCL COMBINED: Vision embeddings gathered via NCCL, shape {vision_embeddings.shape}")
    
    # Step 3: Run LLM on ALL actors with FULL embeddings (TP with all-reduce)
    llm_refs = [
        actors[i].forward_llm.remote(full_vision_outputs[i])
        for i in range(num_actors)
    ]
    llm_outputs = ray.get(llm_refs)
    
    # All TP ranks should have identical output after all-reduce
    llm_output = llm_outputs[0]
    logger.info(f"NCCL COMBINED: LLM output shape {llm_output.shape}")
    
    # Cleanup distributed before killing actors
    cleanup_refs = [actor.cleanup_distributed.remote() for actor in actors]
    ray.get(cleanup_refs)
    
    for actor in actors:
        ray.kill(actor)
    
    import time
    time.sleep(1)  # Allow NCCL resources to be released
    
    return vision_embeddings, llm_output


def run_nccl_disaggregated_4gpu(
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    model_name: str,
    vision_dp: int,
    llm_tp: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run disaggregated approach with NCCL across 4 GPUs.
    
    Vision encoders on GPU 0, 1 (DP)
    LLM actors on GPU 2, 3 (TP)
    
    Uses:
    - NCCL all_gather among vision actors
    - NCCL P2P isend/irecv from vision actor 0 to all LLM actors
    
    Total GPUs: 4 (vision_dp + llm_tp)
    
    Args:
        pixel_values: Input pixel values
        grid_thw: Grid dimensions
        model_name: Model name or path
        vision_dp: Number of vision DP actors (on GPU 0, 1, ...)
        llm_tp: Number of LLM TP actors (on GPU vision_dp, vision_dp+1, ...)
    
    Returns:
        Tuple of (vision_embeddings, llm_output)
    """
    from qwen3_vl_vision_actor import create_vision_actor_class
    from qwen3_vl_llm_actor import create_llm_actor_class
    
    total_world_size = vision_dp + llm_tp
    
    logger.info(
        f"NCCL DISAGGREGATED 4GPU: "
        f"{vision_dp} vision actors (DP) + {llm_tp} LLM actors (TP)"
    )
    logger.info(f"  Vision GPUs: 0 to {vision_dp - 1}")
    logger.info(f"  LLM GPUs: {vision_dp} to {vision_dp + llm_tp - 1}")
    logger.info(f"  Total GPUs: {total_world_size}")
    logger.info(f"  Using NCCL all_gather for vision embeddings")
    logger.info(f"  Using NCCL P2P for vision-to-LLM transfer")
    
    # Create vision actors with DP size
    VisionActorClass = create_vision_actor_class(num_gpus=1, num_cpus=4)
    vision_actors = [
        VisionActorClass.remote(model_name, rank=i, dp_size=vision_dp)
        for i in range(vision_dp)
    ]
    
    # Create LLM actors with global ranks
    LLMActorClass = create_llm_actor_class(num_gpus=1, num_cpus=4)
    llm_actors = [
        LLMActorClass.remote(
            model_name,
            tp_rank=i,
            tp_size=llm_tp,
            global_rank=vision_dp + i,  # Global rank: 2, 3
        )
        for i in range(llm_tp)
    ]
    
    # Get master address from vision rank 0
    master_addr = ray.get(vision_actors[0].get_ip_address.remote())
    master_port = 29508
    
    # Initialize distributed for ALL actors with world_size = 4
    # This allows P2P communication between vision and LLM actors
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
    logger.info(f"NCCL DISAGGREGATED 4GPU: All actors distributed initialized (world_size={total_world_size})")
    
    # Build models
    vision_build_refs = [actor.build_model.remote() for actor in vision_actors]
    llm_build_refs = [actor.build_model.remote() for actor in llm_actors]
    ray.get(vision_build_refs + llm_build_refs)
    logger.info(f"NCCL DISAGGREGATED 4GPU: All models built")
    
    # Split batch across vision actors (DP)
    num_images = grid_thw.shape[0]
    images_per_actor = num_images // vision_dp
    extra = num_images % vision_dp
    patches_per_image = grid_thw[0, 0].item() * grid_thw[0, 1].item() * grid_thw[0, 2].item()
    
    # Step 1: Run vision forward on each vision actor
    vision_refs = []
    pixel_offset = 0
    image_offset = 0
    
    for i in range(vision_dp):
        num_imgs = images_per_actor + (1 if i < extra else 0)
        num_patches = num_imgs * patches_per_image
        
        chunk_pixels = pixel_values[pixel_offset:pixel_offset + num_patches]
        chunk_grid = grid_thw[image_offset:image_offset + num_imgs]
        
        ref = vision_actors[i].forward.remote(chunk_pixels, chunk_grid)
        vision_refs.append(ref)
        
        pixel_offset += num_patches
        image_offset += num_imgs
    
    # Get local vision embeddings
    local_vision_outputs = ray.get(vision_refs)
    logger.info(f"NCCL DISAGGREGATED 4GPU: Vision forward complete")
    
    # Step 2: NCCL all_gather among vision actors
    if vision_dp > 1:
        all_gather_refs = [
            vision_actors[i].all_gather_embeddings.remote(local_vision_outputs[i])
            for i in range(vision_dp)
        ]
        full_vision_outputs = ray.get(all_gather_refs)
        vision_embeddings = full_vision_outputs[0]
    else:
        vision_embeddings = local_vision_outputs[0]
    logger.info(f"NCCL DISAGGREGATED 4GPU: Vision embeddings gathered, shape {vision_embeddings.shape}")
    
    # Step 3: Vision actor 0 sends embeddings to ALL LLM actors via NCCL P2P
    embedding_shape = (vision_embeddings.shape[0], vision_embeddings.shape[1])
    
    # Vision actor 0 sends to each LLM actor, LLM actors receive simultaneously
    send_refs = []
    recv_refs = []
    
    for i in range(llm_tp):
        dst_rank = vision_dp + i  # LLM global ranks start at vision_dp
        send_refs.append(vision_actors[0].send_embeddings_p2p.remote(vision_embeddings, dst_rank))
        recv_refs.append(llm_actors[i].recv_embeddings_p2p.remote(embedding_shape, src_rank=0))
    
    # Wait for all P2P transfers to complete and get received embeddings
    ray.get(send_refs)  # Wait for sends to complete
    received_embeddings = ray.get(recv_refs)  # Get received embeddings
    logger.info(f"NCCL DISAGGREGATED 4GPU: Embeddings transferred via NCCL P2P")
    
    # Step 4: Run LLM forward on all TP ranks
    llm_refs = [
        actor.forward.remote(received_embeddings[i])
        for i, actor in enumerate(llm_actors)
    ]
    llm_outputs = ray.get(llm_refs)
    
    # All TP ranks should have identical output after all-reduce
    llm_output = llm_outputs[0]
    logger.info(f"NCCL DISAGGREGATED 4GPU: LLM output shape {llm_output.shape}")
    
    # Cleanup distributed before killing actors
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
    
    import time
    time.sleep(1)  # Allow NCCL resources to be released
    
    return vision_embeddings, llm_output


def verify_case4_configurable_dp_tp(
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    model_name: str,
    rtol: float,
    atol: float,
) -> bool:
    """
    Case 4: Configurable DP/TP comparison with baseline.
    
    Verify that the baseline (single GPU, no DP, no TP) produces the same
    output as various disaggregated configurations with different DP and TP.
    
    All configurations use NCCL all_gather for vision embeddings.
    
    GPU Placement for disaggregated:
    - Vision on GPUs 0 to (vision_dp - 1)
    - LLM on GPUs vision_dp to (vision_dp + llm_tp - 1)
    
    Configurations tested:
    - Baseline: Single GPU (DP=1, TP=1) on GPU 0
    - Config 1: Vision DP=2, LLM TP=1 (Vision on GPU 0,1, LLM on GPU 2)
    - Config 2: Vision DP=1, LLM TP=2 (Vision on GPU 0, LLM on GPU 1,2)
    - Config 3: Vision DP=2, LLM TP=2 (Vision on GPU 0,1, LLM on GPU 2,3)
    """
    print("\n" + "=" * 70)
    print("CASE 4: Configurable DP/TP - Compare Baseline with NCCL Disaggregated")
    print("=" * 70)
    print("  Baseline: Single GPU, no DP, no TP (GPU 0)")
    print("  All configs use NCCL all_gather for vision embeddings")
    print("  Configuration 1: Vision DP=2, LLM TP=1 (3 GPUs total)")
    print("  Configuration 2: Vision DP=1, LLM TP=2 (3 GPUs total)")
    print("  Configuration 3: Vision DP=2, LLM TP=2 (4 GPUs total)")
    print("=" * 70)
    
    # Run baseline first
    print("\n--- Running BASELINE (Single GPU) ---")
    baseline_vision, baseline_llm = run_baseline_single_gpu(
        pixel_values, grid_thw, model_name
    )
    
    all_passed = True
    
    # Configuration 1: Vision DP=2, LLM TP=1
    print("\n--- Running Config 1: Vision DP=2, LLM TP=1 (NCCL all_gather) ---")
    config1_vision, config1_llm = run_nccl_disaggregated(
        pixel_values, grid_thw, model_name,
        vision_dp=2, llm_tp=1
    )
    
    print("\n--- COMPARISON: Baseline vs Config 1 ---")
    vision_match_1 = compare_tensors(
        baseline_vision, config1_vision,
        "Config 1 Vision Embeddings",
        rtol=rtol, atol=atol
    )
    llm_match_1 = compare_tensors(
        baseline_llm, config1_llm,
        "Config 1 LLM Output",
        rtol=rtol, atol=atol
    )
    config1_passed = vision_match_1 and llm_match_1
    all_passed = all_passed and config1_passed
    
    # Configuration 2: Vision DP=1, LLM TP=2
    print("\n--- Running Config 2: Vision DP=1, LLM TP=2 ---")
    config2_vision, config2_llm = run_nccl_disaggregated(
        pixel_values, grid_thw, model_name,
        vision_dp=1, llm_tp=2
    )
    
    print("\n--- COMPARISON: Baseline vs Config 2 ---")
    vision_match_2 = compare_tensors(
        baseline_vision, config2_vision,
        "Config 2 Vision Embeddings",
        rtol=rtol, atol=atol
    )
    llm_match_2 = compare_tensors(
        baseline_llm, config2_llm,
        "Config 2 LLM Output",
        rtol=rtol, atol=atol
    )
    config2_passed = vision_match_2 and llm_match_2
    all_passed = all_passed and config2_passed
    
    # Configuration 3: Vision DP=2, LLM TP=2
    print("\n--- Running Config 3: Vision DP=2, LLM TP=2 (NCCL all_gather) ---")
    config3_vision, config3_llm = run_nccl_disaggregated(
        pixel_values, grid_thw, model_name,
        vision_dp=2, llm_tp=2
    )
    
    print("\n--- COMPARISON: Baseline vs Config 3 ---")
    vision_match_3 = compare_tensors(
        baseline_vision, config3_vision,
        "Config 3 Vision Embeddings",
        rtol=rtol, atol=atol
    )
    llm_match_3 = compare_tensors(
        baseline_llm, config3_llm,
        "Config 3 LLM Output",
        rtol=rtol, atol=atol
    )
    config3_passed = vision_match_3 and llm_match_3
    all_passed = all_passed and config3_passed
    
    # Summary
    print("\n--- Case 4 Summary ---")
    print(f"  Config 1 (DP=2, TP=1): {'✓ PASSED' if config1_passed else '✗ FAILED'}")
    print(f"  Config 2 (DP=1, TP=2): {'✓ PASSED' if config2_passed else '✗ FAILED'}")
    print(f"  Config 3 (DP=2, TP=2): {'✓ PASSED' if config3_passed else '✗ FAILED'}")
    
    return all_passed


def verify_case5_nccl_combined(
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    model_name: str,
    num_actors: int,
    rtol: float,
    atol: float,
) -> bool:
    """
    Case 5: Verify NCCL-based combined approach produces same output as baseline.
    
    Combined approach:
    - Vision encoders on GPU 0, 1 (DP)
    - LLM on GPU 0, 1 (TP) - SAME GPUs as vision
    - NCCL all_gather for vision embeddings
    
    Compared against:
    - Baseline: 1 encoder, 1 LLM on single GPU (no DP, no TP)
    """
    print("\n" + "=" * 70)
    print("CASE 5: NCCL-based Combined vs Baseline")
    print("=" * 70)
    print(f"  Baseline: Single GPU, no DP, no TP")
    print(f"  NCCL Combined: {num_actors} actors on GPU 0-{num_actors-1}")
    print(f"    - Vision DP={num_actors} (each processes portion of batch)")
    print(f"    - NCCL all_gather (each actor gets full embeddings)")
    print(f"    - LLM TP={num_actors} (all-reduce on outputs)")
    print(f"    - Vision and LLM on SAME GPUs")
    print("=" * 70)
    
    # Run baseline first
    print("\n--- Running BASELINE (Single GPU) ---")
    baseline_vision, baseline_llm = run_baseline_single_gpu(
        pixel_values, grid_thw, model_name
    )
    
    # Run NCCL combined
    print(f"\n--- Running NCCL Combined: {num_actors} actors ---")
    nccl_vision, nccl_llm = run_nccl_combined(
        pixel_values, grid_thw, model_name,
        num_actors=num_actors
    )
    
    # Compare
    print("\n--- COMPARISON ---")
    vision_match = compare_tensors(
        baseline_vision, nccl_vision,
        "Vision Embeddings",
        rtol=rtol, atol=atol
    )
    llm_match = compare_tensors(
        baseline_llm, nccl_llm,
        "LLM Output",
        rtol=rtol, atol=atol
    )
    
    return vision_match and llm_match


def verify_case6_nccl_combined_vs_disaggregated(
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    model_name: str,
    rtol: float,
    atol: float,
) -> bool:
    """
    Case 6: Compare NCCL Combined (GPU 0,1) vs NCCL Disaggregated (GPU 0,1 + 2,3).
    
    Combined (Case 5 style):
    - Vision encoders on GPU 0, 1 (DP)
    - LLM on GPU 0, 1 (TP) - SAME GPUs as vision
    - NCCL all_gather for vision embeddings
    
    Disaggregated:
    - Vision encoders on GPU 0, 1 (DP)
    - LLM on GPU 2, 3 (TP) - DIFFERENT GPUs
    - NCCL all_gather for vision, then send to LLM via Ray
    
    Both should produce identical outputs.
    """
    print("\n" + "=" * 70)
    print("CASE 6: NCCL Combined (GPU 0,1) vs Disaggregated (GPU 0,1 + 2,3)")
    print("=" * 70)
    print("  Combined (2 GPUs):")
    print("    - Vision DP=2 on GPU 0,1")
    print("    - LLM TP=2 on GPU 0,1 (SAME GPUs)")
    print("    - NCCL all_gather for vision embeddings")
    print("  Disaggregated (4 GPUs):")
    print("    - Vision DP=2 on GPU 0,1")
    print("    - LLM TP=2 on GPU 2,3 (DIFFERENT GPUs)")
    print("    - NCCL all_gather for vision, then pass to LLM via Ray")
    print("=" * 70)
    
    # Run Combined (Case 5 style) - GPU 0,1
    print("\n--- Running COMBINED (Vision + LLM on GPU 0,1) ---")
    combined_vision, combined_llm = run_nccl_combined(
        pixel_values, grid_thw, model_name,
        num_actors=2
    )
    
    # Run Disaggregated - Vision GPU 0,1, LLM GPU 2,3
    print("\n--- Running DISAGGREGATED (Vision GPU 0,1, LLM GPU 2,3) ---")
    disagg_vision, disagg_llm = run_nccl_disaggregated_4gpu(
        pixel_values, grid_thw, model_name,
        vision_dp=2, llm_tp=2
    )
    
    # Compare
    print("\n--- COMPARISON ---")
    vision_match = compare_tensors(
        combined_vision, disagg_vision,
        "Vision Embeddings",
        rtol=rtol, atol=atol
    )
    llm_match = compare_tensors(
        combined_llm, disagg_llm,
        "LLM Output",
        rtol=rtol, atol=atol
    )
    
    return vision_match and llm_match


def main():
    parser = argparse.ArgumentParser(
        description="Verify disaggregated vs combined approach equivalence"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="Model name or path",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=4,
        help="Number of images in the batch",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-4,
        help="Relative tolerance for comparison",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-4,
        help="Absolute tolerance for comparison",
    )
    parser.add_argument(
        "--case",
        type=int,
        choices=[4, 5, 6],
        default=None,
        help="Run only specific case (4, 5, or 6). Default: run all",
    )
    parser.add_argument(
        "--vision_dp",
        type=int,
        default=None,
        help="Vision DP size for custom configuration (requires --llm_tp)",
    )
    parser.add_argument(
        "--llm_tp",
        type=int,
        default=None,
        help="LLM TP size for custom configuration (requires --vision_dp)",
    )
    args = parser.parse_args()
    
    # Ensure num_images is divisible by 2 (for DP=2 cases)
    num_dp_actors = 2
    if args.num_images % num_dp_actors != 0:
        args.num_images = (args.num_images // num_dp_actors + 1) * num_dp_actors
        logger.warning(f"Adjusted num_images to {args.num_images} for even split")
    
    print("=" * 70)
    print("Verification: Disaggregated vs Combined Approaches")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Number of images: {args.num_images}")
    print(f"Tolerance: rtol={args.rtol}, atol={args.atol}")
    print(f"GPU constraint: 4 GPUs total")
    print(f"All cases use NCCL all_gather for vision embeddings")
    print("=" * 70)
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init()
    
    # Create dummy batch
    logger.info("Creating dummy batch...")
    pixel_values, grid_thw = create_dummy_batch(args.num_images)
    logger.info(f"Batch created: pixel_values={pixel_values.shape}, grid_thw={grid_thw.shape}")
    
    results = {}
    
    # Handle custom configuration if provided
    if args.vision_dp is not None and args.llm_tp is not None:
        print("\n" + "=" * 70)
        print(f"CUSTOM CONFIG: Vision DP={args.vision_dp}, LLM TP={args.llm_tp}")
        print("=" * 70)
        
        total_gpus = args.vision_dp + args.llm_tp
        if total_gpus > 4:
            print(f"ERROR: Total GPUs required ({total_gpus}) exceeds available (4)")
            return 1
        
        print(f"  Total GPUs: {total_gpus}")
        print(f"  Vision GPUs: 0 to {args.vision_dp - 1}")
        print(f"  LLM GPUs: {args.vision_dp} to {args.vision_dp + args.llm_tp - 1}")
        print(f"  Using NCCL all_gather for vision embeddings")
        
        print("\n--- Running BASELINE (Single GPU) ---")
        baseline_vision, baseline_llm = run_baseline_single_gpu(
            pixel_values, grid_thw, args.model
        )
        
        print(f"\n--- Running NCCL Disaggregated: DP={args.vision_dp}, TP={args.llm_tp} ---")
        config_vision, config_llm = run_nccl_disaggregated(
            pixel_values, grid_thw, args.model,
            vision_dp=args.vision_dp, llm_tp=args.llm_tp
        )
        
        print("\n--- COMPARISON ---")
        vision_match = compare_tensors(
            baseline_vision, config_vision,
            "Vision Embeddings",
            rtol=args.rtol, atol=args.atol
        )
        llm_match = compare_tensors(
            baseline_llm, config_llm,
            "LLM Output",
            rtol=args.rtol, atol=args.atol
        )
        
        results["custom"] = vision_match and llm_match
    else:
        # Case 4: Configurable DP/TP comparison with baseline (using NCCL all_gather)
        if args.case is None or args.case == 4:
            results["case4"] = verify_case4_configurable_dp_tp(
                pixel_values, grid_thw, args.model,
                rtol=args.rtol, atol=args.atol
            )
        
        # Case 5: NCCL-based combined (all_gather via NCCL, same GPUs for vision and LLM)
        if args.case is None or args.case == 5:
            results["case5"] = verify_case5_nccl_combined(
                pixel_values, grid_thw, args.model,
                num_actors=2,  # 2 GPUs: Vision DP=2, LLM TP=2 on same GPUs
                rtol=args.rtol, atol=args.atol
            )
        
        # Case 6: Compare combined (GPU 0,1) vs disaggregated (GPU 0,1 + 2,3)
        if args.case is None or args.case == 6:
            results["case6"] = verify_case6_nccl_combined_vs_disaggregated(
                pixel_values, grid_thw, args.model,
                rtol=args.rtol, atol=args.atol
            )
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    
    all_passed = True
    for case_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {case_name}: {status}")
        all_passed = all_passed and passed
    
    print("=" * 70)
    if all_passed:
        print("✓ ALL TESTS PASSED: Disaggregated and Combined produce identical outputs!")
    else:
        print("✗ SOME TESTS FAILED: Check the logs above for details.")
    print("=" * 70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
