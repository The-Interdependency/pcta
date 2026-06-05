"""Layer-2 tensor objects: the opaque circle carrier and the seed it produces.

pcta consumes **circle-tensors** (layer-1 / `pcna` output — themselves tensors)
and organizes 7 of them into a **seed**, which is itself a tensor. Both objects
are *structural scaffold*: their geometry routes, it does not learn. Per the
stack map, back-propagation lives only in layer 1 (`pcna`); nothing here ever
appears on an autodiff tape, so ``requires_grad`` is always ``False``.

The circle's *internal* structure (7 tensors per circle, the {7/2} step) is
layer-1's business; pcta treats a circle as an **opaque payload host** with a
stable identity and an anchor position, and never inspects or mutates it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class CircleTensor:
    """An opaque layer-1 circle, hosted as a payload by a seed.

    `payload` is whatever layer 1 produced (a tensor / weights handle); pcta
    carries it losslessly and never reads into it. `anchor` is the circle's
    position within its seed (assigned by heptagram routing); `identity` is a
    stable tag for provenance.
    """

    payload: Any
    anchor: int = 0
    identity: Optional[str] = None

    @property
    def requires_grad(self) -> bool:
        # Geometry is non-differentiable scaffold; gradients live in pcna.
        return False


class Seed:
    """A seed tensor: a heptagram grouping of `CIRCLES_PER_SEED` circle-tensors.

    The seed is itself a tensor (it can be hosted by a layer-3 core exactly as a
    circle is hosted here). Its geometry — the anchor visitation order — is
    frozen structural scaffold produced by ``compose_seed``; it does not learn.
    """

    __slots__ = ("circles", "routing_step", "anchor_order", "identity")

    def __init__(
        self,
        circles: Sequence[CircleTensor],
        routing_step: int,
        anchor_order: Tuple[int, ...],
        identity: Optional[str] = None,
    ) -> None:
        self.circles: List[CircleTensor] = list(circles)
        self.routing_step = routing_step
        self.anchor_order = anchor_order
        self.identity = identity

    @property
    def requires_grad(self) -> bool:
        return False

    @property
    def n_circles(self) -> int:
        return len(self.circles)

    def at(self, anchor: int) -> CircleTensor:
        """Retrieve the circle hosted at ``anchor`` (lossless round-trip)."""
        for c in self.circles:
            if c.anchor == anchor:
                return c
        raise KeyError(anchor)

    def circle_tensors(self) -> List[CircleTensor]:
        """The hosted circles in heptagram anchor order."""
        return [self.at(a) for a in self.anchor_order]

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        ident = f" {self.identity!r}" if self.identity else ""
        return f"<Seed{ident} circles={self.n_circles} order={self.anchor_order}>"


@dataclass(frozen=True)
class SeedMotion:
    """The structural **motion** a seed hands upward (to layer 3 / ZFAE).

    "Motion" is described here by *role* — the structural arrangement a seed
    contributes — and its formal definition is `hmmm` in the stack canon (do not
    over-specify it). This carrier captures the observable structure only: the
    seed's identity and the heptagram order its circles were routed in. It holds
    no weights and no gradient (those are pcna's `weights`, a separate channel).
    """

    seed_identity: Optional[str]
    routing_step: int
    anchor_order: Tuple[int, ...]
    circle_identities: Tuple[Optional[str], ...] = field(default_factory=tuple)
