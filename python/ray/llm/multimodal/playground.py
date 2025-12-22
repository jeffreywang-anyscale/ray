import ray

ray.init()

import torch
from ray.experimental.collective import create_collective_group

