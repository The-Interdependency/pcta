"""pcta — Prime-tensor stack **layer 2**: circles -> seeds.

This is the greenfield seed layer of The Interdependency's prime-tensor compute
family. It takes layer-1 (`pcna`) circle-tensors and organizes **7 circles into
a seed** (the seed is itself a tensor), producing structural **motion** that the
inference cap (`zfae`, runtime in `a0`) consumes alongside pcna's trained
weights.

Boundaries (canonical map: `The-Interdependency/interdependent-lib :
docs/prime-tensor-stack.md` — cited, not imported):
  - composition is **structural / non-differentiable**; back-propagation lives
    only in layer 1 (`pcna`). Nothing here carries a gradient.
  - naming another repo's terms transfers **no** theorem / proof / empirical
    status. The coherence-prime rule is *mirrored*, never imported.

`hmmm` (unresolved — do not encode as fact): the PCTA acronym expansion,
seeds-per-core (a layer-3 concern), and the formal definition of "motion".
"""
from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Erin Patrick Spencer <wayseer@interdependentway.org>"
__license__ = "AGPL-3.0-or-later"

from .constants import (
    CIRCLES_PER_SEED,
    HEPTAGRAM_VERTICES,
    SEED_ROUTING_STEP,
    coherence_primes_up_to,
    is_coherence_prime,
    nth_coherence_prime,
)
from .compose import (
    build_seed,
    compose_seed,
    heptagram_order,
    seed_motion,
)
from .tensor import CircleTensor, Seed, SeedMotion

__all__ = [
    "__version__",
    # objects
    "CircleTensor",
    "Seed",
    "SeedMotion",
    # composition
    "compose_seed",
    "build_seed",
    "seed_motion",
    "heptagram_order",
    # constants / guard
    "CIRCLES_PER_SEED",
    "HEPTAGRAM_VERTICES",
    "SEED_ROUTING_STEP",
    "is_coherence_prime",
    "coherence_primes_up_to",
    "nth_coherence_prime",
]
