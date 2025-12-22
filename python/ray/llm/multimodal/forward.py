import logging
import ray
import torch

from actor_group import ActorGroup
from model_actor import QwenVisionActor, QwenTextActor

logger = logging.getLogger(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)


def setup_actors(num_vision_actors=2, num_text_actors=1, collocate=False):
    """
    Set up vision (DP) and text actor groups.
    
    Args:
        num_vision_actors: Number of vision actors for data parallelism (default: 2 for 2 GPUs)
        num_text_actors: Number of text actors (default: 1, no TP for now)
        collocate: Whether to collocate vision and text actors on same GPUs
    
    Returns:
        Tuple of (vision_group, text_group)
    """
    # Vision actors: Data Parallelism (each processes different mini-batch)
    # Using 2 actors for 2 GPUs
    vision_group = ActorGroup(
        QwenVisionActor,
        num_actors=num_vision_actors,
        num_cpus=4,
        num_gpus=1,
        collocate=collocate
    )
    
    # Text actor(s): No TP for now, just single actor processing full batch
    text_group = ActorGroup(
        QwenTextActor,
        num_actors=num_text_actors,
        num_cpus=4,
        num_gpus=1,
        collocate=collocate,
        placement_group_handle=vision_group.placement_group if collocate else None
    )
    
    # Build models on all actors
    print("Building vision models...")
    vision_group.execute_all("build_model")
    print("Building text models...")
    text_group.execute_all("build_model")
    
    print(f"Set up {num_vision_actors} vision actors (DP) and {num_text_actors} text actor(s)")
    
    return vision_group, text_group


def forward(vision_group, text_group, batch):
    """
    Execute forward pass with vision DP.
    
    Args:
        vision_group: Vision actor group (DP)
        text_group: Text actor group
        batch: Full batch tensor [batch_size, ...] to be split into mini-batches
    
    Returns:
        Text outputs from text actor(s)
    """
    # Split batch into mini-batches for data parallelism
    # Each vision actor processes one mini-batch
    batch_size = batch.shape[0]
    num_vision_actors = vision_group.num_actors
    
    # Split batch into mini-batches
    mini_batch_size = batch_size // num_vision_actors
    mini_batches = []
    for i in range(num_vision_actors):
        start_idx = i * mini_batch_size
        end_idx = start_idx + mini_batch_size if i < num_vision_actors - 1 else batch_size
        mini_batches.append(batch[start_idx:end_idx])
    
    print(f"Split batch of size {batch_size} into {len(mini_batches)} mini-batches")
    
    # Vision forward pass: Data Parallelism
    # Each vision actor processes a different mini-batch
    print("Running vision forward pass...")
    vision_refs = vision_group.execute_all_async("forward", mini_batches)
    
    # Text forward pass: Single actor processes concatenated vision outputs
    print("Running text forward pass...")
    text_refs = text_group.execute_all_async("forward", vision_refs)
    
    # Wait for all results
    text_outputs = ray.get(text_refs)
    
    print(f"Forward pass completed. Text outputs: {len(text_outputs)}")
    
    return text_outputs


# Example usage
if __name__ == "__main__":
    # Initialize Ray
    if not ray.is_initialized():
        ray.init()
    
    print("=" * 50)
    print("Multimodal Forward Pass Demo")
    print("=" * 50)
    
    # Set up actors: 2 vision actors (DP), 1 text actor (no TP)
    # This uses 3 GPUs total (2 for vision, 1 for text)
    vision_group, text_group = setup_actors(num_vision_actors=2, num_text_actors=1)
    
    # Create dummy batch
    batch_size = 8  # Total batch size
    hidden_size = 10
    dummy_batch = torch.randn(batch_size, hidden_size)
    print(f"\nInput batch shape: {dummy_batch.shape}")
    
    # Execute forward pass
    outputs = forward(vision_group, text_group, dummy_batch)
    
    print("\n" + "=" * 50)
    print("Results:")
    print("=" * 50)
    print(f"Got {len(outputs)} text output(s)")
    for i, output in enumerate(outputs):
        print(f"  Text actor {i} output shape: {output.shape}")
