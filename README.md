# pcta — prime-tensor stack layer 2 (circles → seeds)

`pcta` is the **seed layer** of [The Interdependency](https://github.com/The-Interdependency)'s
prime-tensor compute family. It takes layer-1 (`pcna`) **circle-tensors** and
organizes **7 circles into a seed** — and the seed is itself a tensor. The
structural arrangement it produces ("motion") is consumed downstream by the
inference cap (`zfae`, runtime in `a0`) together with pcna's trained weights.

```
PCNA (tensors + backprop) ─► weights ┐
PCTA (circles → seeds) ─┐            ├─► ZFAE (inference) ─► output
PTCA (seeds → core)  ───┴─► motion ──┘
PCEA — guardian: seals the weights / state (orthogonal; not a layer)
```

The canonical role-and-boundary map is
`The-Interdependency/interdependent-lib : docs/prime-tensor-stack.md`. This repo
**cites** it and does not import it; naming another repo's terms transfers **no**
theorem / proof / empirical status.

## Install

```bash
pip install -e .            # zero runtime dependencies (stdlib only)
pip install -e ".[dev]"     # adds pytest
```

Requires Python ≥ 3.9.

## Use

```python
from pcta import build_seed, compose_seed, CircleTensor, seed_motion

# Wrap seven layer-1 circle outputs (opaque tensors / weight handles) ...
seed = build_seed([f"circle_weights_{i}" for i in range(7)], identity="seed:13")

# ... or compose explicit CircleTensors:
circles = [CircleTensor(payload=w, identity=f"c{i}") for i, w in enumerate(ws)]
seed = compose_seed(circles)               # {7/3} heptagram order; 7 circles required

seed.n_circles            # 7
seed.anchor_order         # (0, 3, 6, 2, 5, 1, 4)  — the {7/3} star order
seed.at(0).payload        # lossless: the original circle payload

motion = seed_motion(seed)   # structural motion handed upward (no weights/grads)
```

## Boundaries

- **Structural, non-differentiable.** Composition is the `⊠` operator: it
  assembles circles into seeds. Back-propagation lives **only** in layer 1
  (`pcna`); nothing here carries a gradient (`requires_grad` is always `False`).
- **Opaque circles.** A circle's internal structure (7 tensors per circle, the
  `{7/2}` step) is layer-1's business. pcta treats a circle as an opaque payload
  host and never inspects or mutates it.
- **Coherence-prime rule is mirrored, not imported.** `is_coherence_prime`
  reproduces `interdependent_lib.coherence_primes` exactly (including the
  `p=4373` regression). Importing the aggregator would invert the dependency
  graph.

## `hmmm` (unresolved — do not encode as fact)

- the **PCTA acronym expansion** (no agreed expansion across the org; an earlier
  README draft read "prime circle tensor architecture" — not canonical);
- **seeds per core** (a layer-3 / `ptca` concern; the stack map lists it `hmmm`);
- the **formal definition of "motion"** (described by role only).

## License

AGPL-3.0-or-later, dual-licensed with a commercial option. See `LICENSE` and
`LICENSE-COMMERCIAL.md`. Copyright (c) 2026 Erin Patrick Spencer.
