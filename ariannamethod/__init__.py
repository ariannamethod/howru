"""
ariannamethod — the notorch shim that gives Howru a from-scratch, torch-free trainer.

C line:  notorch.c / notorch.h  (pure-C neural framework — tape autograd, RRPRAM,
                                 Janus-echo primitives, Chuck)
Python:  notorch_nn.py          (ctypes; HowruEngine builds Howru's transformer)
         chuck.py               (the self-aware optimizer, nt_tape_chuck_step)
"""

from .notorch_nn import Tensor, HowruEngine, softmax, multinomial, seed
from .chuck import ChuckOptimizer

__all__ = ['Tensor', 'HowruEngine', 'softmax', 'multinomial', 'seed', 'ChuckOptimizer']
