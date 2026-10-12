"""Small non-reentrant checkpoint boundaries; never wrap DDP itself."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import random
import numpy as np
from contextlib import nullcontext


class _ReplayRandomState:
    """Reusable Python/NumPy RNG context for every checkpoint recomputation.

    Adaptive GAN calls autograd.grad(retain_graph=True) before total.backward().
    PyTorch reuses the same context object on those separate traversals, so a
    one-shot @contextmanager generator is unsuitable here. Torch RNG is handled
    separately by checkpoint(preserve_rng_state=True).
    """
    def __init__(self):
        self.python_state = random.getstate()
        self.numpy_state = np.random.get_state()
        self.previous_states = []

    def __enter__(self):
        self.previous_states.append((random.getstate(), np.random.get_state()))
        random.setstate(self.python_state)
        np.random.set_state(self.numpy_state)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        python_state, numpy_state = self.previous_states.pop()
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        # Restore state even on checkpoint early-stop or a real exception.
        return False

def run_checkpoint(function,*args,enabled=False):
    if enabled and torch.is_grad_enabled():
        return checkpoint(function,*args,use_reentrant=False,preserve_rng_state=True)
    return function(*args)

def block(module,x,enabled=False):
    # Replaying BatchNorm would update its running statistics twice. The
    # supplied AE configs use GroupNorm; leave custom stateful blocks eager.
    stateful=any(isinstance(m,nn.modules.batchnorm._BatchNorm) and m.training
                 and m.track_running_stats for m in module.modules())
    return run_checkpoint(module,x,enabled=enabled and module.training and not stateful)

def loss_checkpoint(function,*args,enabled=False):
    """Also replay Python/NumPy random slice selection used by external losses."""
    if not enabled or not torch.is_grad_enabled():return function(*args)
    replay = _ReplayRandomState()
    return checkpoint(function,*args,use_reentrant=False,preserve_rng_state=True,
                      context_fn=lambda:(nullcontext(),replay))
