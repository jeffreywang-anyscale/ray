import logging
import ray
from ray.util.placement_group import PlacementGroupSchedulingStrategy, placement_group

logger = logging.getLogger(__name__)


class ActorGroup:
    """Actor group for managing multiple Ray actors with placement group support."""
    
    def __init__(
        self,
        actor_cls,
        num_actors: int,
        num_cpus: int = 4,
        num_gpus: int = 1,
        collocate: bool = False,
        placement_group_handle=None,
        **actor_kwargs
    ):
        """Initialize actor group.
        
        Args:
            actor_cls: Ray actor class to instantiate
            num_actors: Number of actors to create
            num_cpus: CPUs per actor
            num_gpus: GPUs per actor (fractional if collocating)
            collocate: Whether actors should be collocated with another group on same GPUs
            placement_group_handle: Existing placement group to use (for collocation)
            **actor_kwargs: Additional kwargs to pass to actor __init__
        """
        self.num_actors = num_actors
        self.collocate = collocate
        
        # Calculate GPU allocation per actor
        # When collocating, use fractional GPUs (e.g., 0.5 per actor)
        gpus_per_actor = num_gpus / 2 if collocate else num_gpus
        
        logger.info(f"Creating ActorGroup with {num_actors} actors")
        logger.info(f"Collocation: {collocate}, GPUs per actor: {gpus_per_actor}")
        
        # Support both plain Python classes and already-remote Ray actor classes
        if hasattr(actor_cls, "remote"):
            remote_actor_cls = actor_cls
        else:
            # Create remote actor class with base resource requirements
            remote_actor_cls = ray.remote(num_cpus=num_cpus, num_gpus=gpus_per_actor)(actor_cls)
        
        # Create or reuse placement group for collocation
        if collocate:
            if placement_group_handle is None:
                # Create new placement group
                bundles = [{"GPU": num_gpus, "CPU": num_cpus} for _ in range(num_actors)]
                self.placement_group = placement_group(bundles, strategy="PACK")
                ray.get(self.placement_group.ready())
                logger.info(f"Created new placement group for {num_actors} actors")
            else:
                # Reuse existing placement group (for second model group)
                self.placement_group = placement_group_handle
                logger.info("Reusing existing placement group")
        else:
            self.placement_group = None
        
        # Create actors
        self._actors = []
        for i in range(num_actors):
            if self.placement_group is not None:
                # Use placement group scheduling
                actor = remote_actor_cls.options(
                    num_cpus=num_cpus / 2 if collocate else num_cpus,
                    num_gpus=gpus_per_actor,
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=self.placement_group,
                        placement_group_bundle_index=i,
                    ),
                ).remote(rank=i, **actor_kwargs)
            else:
                # Standard scheduling (Ray handles placement automatically)
                actor = remote_actor_cls.options(
                    num_cpus=num_cpus,
                    num_gpus=gpus_per_actor,
                ).remote(rank=i, **actor_kwargs)
            self._actors.append(actor)
        
        logger.info(f"Created ActorGroup with {num_actors} actors")
    
    def execute_all_async(self, method_name: str, *args, **kwargs):
        """Execute a method on all actors asynchronously.
        
        Args:
            method_name: Name of the method to execute
            *args: Positional arguments (if list, distributes across actors)
            **kwargs: Keyword arguments (if list, distributes across actors)
        
        Returns:
            List of remote object references
        """
        # Check if args/kwargs should be distributed
        if args and isinstance(args[0], list) and len(args[0]) == self.num_actors:
            # Distribute arguments across actors
            results = []
            for i, actor in enumerate(self._actors):
                distributed_args = tuple(arg[i] if isinstance(arg, list) else arg for arg in args)
                distributed_kwargs = {k: (v[i] if isinstance(v, list) else v) for k, v in kwargs.items()}
                method = getattr(actor, method_name)
                results.append(method.remote(*distributed_args, **distributed_kwargs))
            return results
        else:
            # Use same arguments for all actors
            results = []
            for actor in self._actors:
                method = getattr(actor, method_name)
                results.append(method.remote(*args, **kwargs))
            return results
    
    def execute_all(self, method_name: str, *args, **kwargs):
        """Execute a method on all actors synchronously.
        
        Args:
            method_name: Name of the method to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
        
        Returns:
            List of results from all actors
        """
        refs = self.execute_all_async(method_name, *args, **kwargs)
        return ray.get(refs)
