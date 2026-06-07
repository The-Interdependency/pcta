"""pcta — **PCTA: Prime Circled Tensor Architecture** (prime-tensor stack layer 2).

Covers circles carried by UCNS objects and composes them into **seeds**. It
takes layer-1 (`pcna`) circle-tensors and organizes a **variable** number of
circles into a seed (the seed is itself a tensor), producing structural
**motion** that PTCA (layer 3) folds into cores and the inference cap (`zfae`,
runtime in `a0`) ultimately consumes alongside pcna's trained weights.

  PCNA (tensors → circles, backprop) ─► circles ─► PCTA (circles → seeds)
  ─► seeds ─► PTCA (seeds → core) ─► cores ─► a0(zfae) inference

Boundaries (canonical map: `The-Interdependency/interdependent-lib :
docs/prime-tensor-stack.md` — cited, not imported):
  - **Composition counts are variable.** The only invariant is that every
    circle is a tensor and every seed is itself a tensor.
  - composition is **structural / non-differentiable**; back-propagation lives
    only in layer 1 (`pcna`). Nothing here carries a gradient.
  - naming another repo's terms transfers **no** theorem / proof / empirical
    status. The coherence-prime rule is *mirrored*, never imported.

"Motion" (the structural output a seed hands upward) is formally the Fickian
gradient flux ``J = −D ∇φ`` (Fick's first law) — structure diffusing down its
field gradient; structural / non-differentiable. No stack `hmmm` remains.
"""
from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Erin Patrick Spencer <wayseer@interdependentway.org>"
__license__ = "AGPL-3.0-or-later"

from .constants import (
    HEPTAGRAM_VERTICES,
    NOMINAL_CIRCLES_PER_SEED,
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
    "NOMINAL_CIRCLES_PER_SEED",
    "HEPTAGRAM_VERTICES",
    "SEED_ROUTING_STEP",
    "is_coherence_prime",
    "coherence_primes_up_to",
    "nth_coherence_prime",
]
