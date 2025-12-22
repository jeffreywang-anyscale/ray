import logging
import ray
import torch

from model import QwenVisionModel, QwenTextModel

logger = logging.getLogger(__name__)


@ray.remote(num_gpus=1, num_cpus=4)
class QwenVisionActor:
    """Vision actor with data parallelism - each actor processes a different mini-batch.
    
    Note: Actor placement is controlled by Ray's placement groups (via ActorGroup),
    not by torch.device. torch.device("cuda:0") just tells PyTorch which CUDA device
    to use within the actor (usually 0 since each actor gets its own GPU via CUDA_VISIBLE_DEVICES).
    """
    
    def __init__(self, rank: int):
        self.rank = rank
        # torch.device is for PyTorch, not Ray placement
        # Ray placement is controlled by @ray.remote(num_gpus=1) and placement groups
        self.device = torch.device("cuda:0")
        self.model = None
        logger.info(f"Vision actor {rank} initialized")
    
    def _create_model_instance(self):
        """Create the vision encoder model."""
        model = QwenVisionModel()
        model = model.to(self.device)
        model.eval()
        return model
    
    def build_model(self):
        """Build the vision model."""
        self.model = self._create_model_instance()
        logger.info(f"Vision actor {self.rank}: model built")
    
    def forward(self, mini_batch):
        """
        Forward pass on a mini-batch (data parallelism).
        
        Args:
            mini_batch: A single mini-batch tensor [batch_size, ...]
        
        Returns:
            Vision embeddings tensor
        """
        if self.model is None:
            raise RuntimeError(f"Vision actor {self.rank}: model not built. Call build_model() first.")
        
        # Move mini_batch to device
        if isinstance(mini_batch, torch.Tensor):
            mini_batch = mini_batch.to(self.device)
        
        # Forward pass
        with torch.no_grad():
            vision_embeddings = self.model(mini_batch)
        
        logger.debug(f"Vision actor {self.rank}: processed batch shape {mini_batch.shape}, output shape {vision_embeddings.shape}")
        return vision_embeddings


@ray.remote(num_gpus=1, num_cpus=4)
class QwenTextActor:
    """Text actor that processes the full batch from vision actors.
    
    For this prototype:
    - Receives concatenated vision embeddings from all vision actors
    - Runs the text model on the full batch
    
    Note: Actor placement is controlled by Ray's placement groups (via ActorGroup),
    not by torch.device.
    """
    
    def __init__(self, rank: int, **kwargs):
        self.rank = rank
        self.device = torch.device("cuda:0")
        self.model = None
        logger.info(f"Text actor {rank} initialized")
    
    def _create_model_instance(self):
        """Create the text encoder model."""
        model = QwenTextModel()
        model = model.to(self.device)
        return model
    
    def build_model(self):
        """Build the text model for inference."""
        model = self._create_model_instance()
        model.eval()
        self.model = model
        logger.info(f"Text actor {self.rank}: model built")
    
    def forward(self, vision_embeddings_refs):
        """
        Forward pass on full batch from vision actors.
        
        Args:
            vision_embeddings_refs: List of Ray object references to vision embeddings
                                   from all vision actors [ref_0, ref_1, ...]
        
        Returns:
            Text output tensor
        """
        if self.model is None:
            raise RuntimeError(f"Text actor {self.rank}: model not built. Call build_model() first.")
        
        # Get actual tensors from Ray object refs
        vision_tensors = []
        for ref in vision_embeddings_refs:
            if isinstance(ref, ray.ObjectRef):
                emb = ray.get(ref)
            else:
                emb = ref
            vision_tensors.append(emb.to(self.device))
        
        # Concatenate all vision embeddings to form full batch
        full_batch = torch.cat(vision_tensors, dim=0)
        
        logger.info(f"Text actor {self.rank}: processing batch shape {full_batch.shape}")
        
        # Forward pass through model
        with torch.no_grad():
            text_output = self.model(full_batch)
        
        logger.info(f"Text actor {self.rank}: output shape {text_output.shape}")
        return text_output
