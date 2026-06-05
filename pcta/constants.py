"""Frozen composition counts, heptagram routing, and the coherence-prime guard.

`pcta` is **layer 2** of the prime-tensor stack (circles -> seeds). The single
source of truth for the stack's role-and-boundary map is
`The-Interdependency/interdependent-lib : docs/prime-tensor-stack.md` — this repo
*cites* it and does not import it. No theorem / proof / empirical status moves
between repos by naming these terms.

What is canonical here (from the stack map):
  - a seed is composed of **7 circles** (`CIRCLES_PER_SEED`), and the seed is
    itself a tensor;
  - the composition is **structural / non-differentiable** — back-propagation
    lives ONLY in layer 1 (`pcna`). pcta organizes; it does not train.

What is deliberately NOT fixed here (marked `hmmm`, per org doctrine — do not
guess):
  - the **acronym expansion** of "PCTA" (no agreed expansion across the org);
  - **seeds per core** (layer 3 / `ptca`'s concern; `prime_core` uses 157
    experimentally, the stack map lists it as `hmmm`);
  - the formal definition of "motion" (the structural output handed to ZFAE).
"""
from __future__ import annotations

from typing import List

# === MODULE_BUILD ===
# id: pcta_constants
#   module_name: constants
#   module_kind: engine
#   summary: frozen layer-2 composition counts, heptagram routing step, and the recursive coherence-prime guard
#   owner: Erin Patrick Spencer
#   public_surface: CIRCLES_PER_SEED, SEED_ROUTING_STEP, HEPTAGRAM_VERTICES, is_coherence_prime, coherence_primes_up_to, nth_coherence_prime
#   internal_surface: _build_coherence_up_to, _is_prime, _prime_factors
#   auth_boundary: none
#   storage_boundary: none
#   network_boundary: none
#   user_data_boundary: none
#   admin_only: false
#   tests: tests.test_constants
#   rollout: default_enabled (imported by pcta.compose via pcta.__init__)
#   rollback: none (greenfield module; revert the file)
#   requires: coherence_primes (mirrored from interdependent_lib, NOT imported — importing the aggregator would invert the dependency graph)
#   since: 2026-06-05 (greenfield scaffold of the layer-2 seed package)
#   unresolved: PCTA acronym expansion = hmmm; seeds-per-core = hmmm (layer-3 concern); formal definition of "motion" = hmmm
# === END MODULE_BUILD ===

# --- canonical layer-2 composition counts (stack map, row 2) -----------------
CIRCLES_PER_SEED: int = 7   # 7 circles per seed; the seed is itself a tensor
HEPTAGRAM_VERTICES: int = 7  # heptagram routing operates on 7 vertices

# --- heptagram routing step --------------------------------------------------
# {7/3} composes circles -> seed. This mirrors `PTCA/prime_core`'s
# SEED_ROUTING_STEP exactly; the circle-level step ({7/2}, tensors -> circle)
# belongs to layer 1 (`pcna`), not here.
SEED_ROUTING_STEP: int = 3

# --- coherence-prime ladder (consciousness primes) ---------------------------
# The membership rule is *recursive*: a prime's kernel (p-1)//4 must be
# square-free and factor only into earlier coherence primes (genealogical
# ancestry), not into a fixed pre-listed universe. A frozen-universe cap silently
# disagrees with this rule for the first time at p=4373, whose kernel 1093 is
# itself a coherence prime above any small cap.
#
# CANON: the single source of truth for this sequence is
#   interdependent_lib.coherence_primes  (The-Interdependency/interdependent-lib)
# `pcta` cannot import it — interdependent-lib optionally depends on the leaf
# libraries, so importing the aggregator here would invert the dependency graph.
# The recursive algorithm is therefore mirrored verbatim; behaviour MUST match
# canon, whose shared test oracle includes the p=4373 regression.
_COHERENCE_BASE = frozenset({3, 5, 7})
_coherence_known: set = set(_COHERENCE_BASE)
_coherence_scanned_to: int = max(_COHERENCE_BASE)


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    d = 2
    while d * d <= n:
        if n % d == 0:
            return False
        d += 1
    return True


def _prime_factors(n: int) -> List[int]:
    factors: List[int] = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors


def _build_coherence_up_to(limit: int) -> None:
    """Admit every coherence prime ``<= limit`` in ascending order so the
    recursive ancestry check can consult the already-admitted set. Idempotent:
    ``_coherence_scanned_to`` guards against re-scanning."""
    global _coherence_scanned_to
    if limit <= _coherence_scanned_to:
        return
    for p in range(_coherence_scanned_to + 1, limit + 1):
        if not _is_prime(p) or p in _COHERENCE_BASE or (p - 1) % 4 != 0:
            continue
        factors = _prime_factors((p - 1) // 4)
        if len(set(factors)) != len(factors):          # square-free
            continue
        if set(factors) <= _coherence_known:           # recursive ancestry
            _coherence_known.add(p)
    _coherence_scanned_to = limit


def is_coherence_prime(p: int) -> bool:
    """Coherence-prime membership test.

    ``p`` is a coherence prime iff either:
      - ``p`` is in the base set ``{3, 5, 7}``, or
      - ``p`` is prime, ``p % 4 == 1``, ``q = (p - 1) // 4`` is square-free, and
        every prime factor of ``q`` is itself a coherence prime (recursive).

    Mirrors ``interdependent_lib.coherence_primes`` exactly — see the CANON note
    above. The ladder begins 3, 5, 7, 13, 29, 53, 61, 157, 349, 421, ...
    """
    if p < 2:
        return False
    if p in _COHERENCE_BASE:
        return True
    if not _is_prime(p) or p % 4 != 1:
        return False
    _build_coherence_up_to(p)
    return p in _coherence_known


def coherence_primes_up_to(limit: int) -> List[int]:
    """Ascending list of coherence primes ``<= limit`` (base set included)."""
    if limit < 0:
        return []
    _build_coherence_up_to(limit)
    return sorted(p for p in _coherence_known if p <= limit)


def nth_coherence_prime(n: int) -> int:
    """The ``n``-th coherence prime, 0-indexed (``nth_coherence_prime(0) == 3``).

    Used to address seeds on the coherence ladder. Note: *how many* seeds a core
    carries (seeds-per-core) is layer-3's concern and is `hmmm` — this helper
    only enumerates the ladder, it does not assert a core size.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    limit = 16
    while True:
        ladder = coherence_primes_up_to(limit)
        if len(ladder) > n:
            return ladder[n]
        limit *= 2
