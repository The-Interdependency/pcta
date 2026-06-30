# ratios: loc_comments=56:36 imports_exports=5:4 calls_definitions=21:4
"""The layer-2 composition operator: circles -> seed.

`compose_seed` is the structural ``⊠`` operator for layer 2. It is **purely
structural**: it grafts circle-tensors into a seed carrier and assigns their
heptagram anchor order. It creates no scalar and registers no autodiff node, so
``∂(⊠)`` never appears on a tape (back-propagation lives only in layer 1,
`pcna`). See `docs`/the stack canon for the boundary.
"""
from __future__ import annotations

from math import gcd
from typing import List, Optional, Sequence

from .constants import HEPTAGRAM_VERTICES, SEED_ROUTING_STEP
from .tensor import CircleTensor, Seed, SeedMotion


def heptagram_order(step: int, n: int = HEPTAGRAM_VERTICES) -> List[int]:
    """Vertex visitation order of the ``{n/step}`` star polygon.

    For ``n = 7``: ``step 2 -> [0,2,4,6,1,3,5]``; ``step 3 -> [0,3,6,2,5,1,4]``.
    Requires ``gcd(step, n) == 1`` so every vertex is visited exactly once.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if gcd(step, n) != 1:
        raise ValueError(
            f"{{{n}/{step}}} is not a single cycle: gcd({step}, {n}) != 1"
        )
    return [(step * i) % n for i in range(n)]


def compose_seed(
    circles: Sequence[CircleTensor],
    *,
    routing_step: int = SEED_ROUTING_STEP,
    identity: Optional[str] = None,
) -> Seed:
    """Compose circle-tensors into a seed (the structural ``⊠`` operator).

    The circle count is **variable** — a seed may carry any number of circles
    (the invariant is only that every circle is a tensor and the seed is itself a
    tensor). Each circle is (re)assigned an ``anchor`` from the
    ``{n/routing_step}`` star-polygon order, preserving input order at the
    assigned positions; for the nominal ``n=7`` case this is the ``{7/3}``
    heptagram. Structural only — no autodiff node is created.
    """
    circles = list(circles)
    n = len(circles)
    if n == 0:
        raise ValueError("cannot compose a seed from zero circles")

    # Apply the {n/step} star-polygon order when it forms a single cycle
    # (gcd(step, n) == 1, e.g. the nominal {7/3} heptagram); otherwise fall back
    # to identity order — the star polygon is undefined when step and n share a
    # factor.
    if gcd(routing_step, n) == 1:
        order = heptagram_order(routing_step, n)
    else:
        order = list(range(n))
    # Assign anchors: the i-th input circle lands at heptagram position order[i].
    anchored: List[CircleTensor] = [
        CircleTensor(payload=c.payload, anchor=order[i], identity=c.identity)
        for i, c in enumerate(circles)
    ]
    return Seed(
        circles=anchored,
        routing_step=routing_step,
        anchor_order=tuple(order),
        identity=identity,
    )


def build_seed(
    payloads: Sequence[object],
    *,
    routing_step: int = SEED_ROUTING_STEP,
    identity: Optional[str] = None,
) -> Seed:
    """Convenience builder: wrap raw circle payloads as ``CircleTensor``s and
    compose them into a seed. ``payloads[i]`` becomes the i-th input circle with
    identity ``"{identity}.c{i}"`` (or ``"c{i}"`` if no seed identity is given).
    Any number of payloads is accepted (composition counts are variable).
    """
    prefix = f"{identity}." if identity else ""
    circles = [
        CircleTensor(payload=p, identity=f"{prefix}c{i}")
        for i, p in enumerate(payloads)
    ]
    return compose_seed(circles, routing_step=routing_step, identity=identity)


def seed_motion(seed: Seed) -> SeedMotion:
    """Extract the structural **motion** a seed hands upward (toward PTCA's core
    composition and ultimately ZFAE inference).

    Returns the observable structure only — identity + star-polygon order — never
    weights or autodiff gradients. Formally, motion is the Fickian gradient flux
    ``J = −D ∇φ`` (Fick's first law) — the seed's composed field diffusing down
    its gradient; this carrier holds the structure that flux rides on.
    """
    return SeedMotion(
        seed_identity=seed.identity,
        routing_step=seed.routing_step,
        anchor_order=seed.anchor_order,
        circle_identities=tuple(c.identity for c in seed.circle_tensors()),
    )
# ratios: loc_comments=56:36 imports_exports=5:4 calls_definitions=21:4
