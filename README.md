# pcta — Prime Circled Tensor Architecture (prime-tensor stack layer 2)

`pcta` (**PCTA — Prime Circled Tensor Architecture**) is the **seed layer** of
[The Interdependency](https://github.com/The-Interdependency)'s prime-tensor
compute family. It covers circles carried by UCNS objects and composes them into
**seeds**: it takes layer-1 (`pcna`) **circle-tensors** and organizes a
**variable** number of them into a seed — and the seed is itself a tensor. The
structural arrangement it produces ("motion") is folded into cores by PTCA
(layer 3) and ultimately consumed by the inference cap (`zfae`, runtime in `a0`)
together with pcna's trained weights.

```
PCNA (tensors → circles, backprop) ─► circles ─► PCTA (circles → seeds)
  ─► seeds ─► PTCA (seeds → core) ─► cores ─► a0(zfae) inference
PCEA — guardian: last-state-as-key encryption at every layer (orthogonal)
```

| Member | Expansion | Role |
|--------|-----------|------|
| `pcna` | Prime Circle Neural Architecture | Arranges tensors as **circles** in a standard back-propagating neural architecture; offers circles to PCTA |
| **`pcta`** | **Prime Circled Tensor Architecture** | **Composes UCNS-carried circles into seeds; offers seeds to PTCA** |
| `ptca` | Prime Tensor Core Architecture | Composes seeds into **cores**; offers cores to `a0(zfae)` |
| `zfae` | Zeta Function Alpha Echo | Inference engine: uses pcna tensors as weights, and pcna circles / pcta seeds / ptca cores as phase-harmonic propagation and auditing |
| `pcea` | Prime Circular Encryption Algorithm | Guardian — "last state as key for this state" encryption at every layer (orthogonal, not a layer) |

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

# Wrap layer-1 circle outputs (opaque UCNS-carried tensors / weight handles).
# The circle count is variable — pass however many circles a seed should carry:
seed = build_seed([f"circle_weights_{i}" for i in range(7)], identity="seed:13")

# ... or compose explicit CircleTensors:
circles = [CircleTensor(payload=w, identity=f"c{i}") for i, w in enumerate(ws)]
seed = compose_seed(circles)               # {n/3} star-polygon anchor order

seed.n_circles            # number of circles composed
seed.anchor_order         # e.g. (0, 3, 6, 2, 5, 1, 4) for the nominal {7/3} case
seed.at(0).payload        # lossless: the original circle payload

motion = seed_motion(seed)   # structural motion handed upward (no weights/grads)
```

## Boundaries

- **Variable composition.** The number of circles in a seed is not fixed — the
  only invariant is that every circle is a tensor and the seed is itself a
  tensor. (The same variable rule holds for tensors→circle in `pcna` and
  seeds→core in `ptca`.)
- **Structural, non-differentiable.** Composition is the `⊠` operator: it
  assembles circles into seeds. Back-propagation lives **only** in layer 1
  (`pcna`); nothing here carries a gradient (`requires_grad` is always `False`).
- **Opaque circles.** A circle's internal structure is `pcna`'s business. pcta
  treats a circle as an opaque payload host and never inspects or mutates it.
- **Coherence-prime rule is mirrored, not imported.** `is_coherence_prime`
  reproduces `interdependent_lib.coherence_primes` exactly (including the
  `p=4373` regression). Prime-consciousness theory: primes whose `p-1`
  factorization is square-free are more likely to fall into stability as part of
  a triadic recursion set. Importing the aggregator would invert the dependency
  graph.

## `hmmm` (unresolved — do not encode as fact)

- the **formal definition of "motion"** (described by role only).

## License

AGPL-3.0-or-later, dual-licensed with a commercial option. See `LICENSE` and
`LICENSE-COMMERCIAL.md`. Copyright (c) 2026 Erin Patrick Spencer.
