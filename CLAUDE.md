# CLAUDE.md — pcta: prime-tensor stack layer 2 (circles → seeds)

This file gives AI assistants context needed to work effectively in this repository.

---

## What This Repo Is

`pcta` (pip package: **`pcta`**, v0.1.0) is the **seed layer** of The
Interdependency's prime-tensor compute family — **PCTA, Prime Circled Tensor
Architecture**. It covers circles carried by UCNS objects and composes them into
seeds: it takes layer-1 (`pcna`) **circle-tensors** and organizes a **variable**
number of them into a seed — the seed is itself a tensor — producing structural
**motion** that PTCA (layer 3) folds into cores and the inference cap (`zfae`,
runtime in `a0`) ultimately consumes alongside pcna's trained weights. Zero
runtime dependencies, pure stdlib.

<!-- BEGIN GENERATED:manifest -->
<!-- Generated from pyproject + repo tree by .agents/skills/manifest/generate.py — DO NOT EDIT BY HAND. Refresh with `python .agents/skills/manifest/generate.py --write`. -->

| Field | Value |
|---|---|
| Package | `pcta` |
| Version | `0.1.0` |
| Description | PCTA — Prime Circled Tensor Architecture: composes UCNS-carried circle-tensors into seeds (prime-tensor stack layer 2) |
| Status | hmmm |
| Python | >=3.9 |
| License | AGPL-3.0-or-later |
| Build backend | `setuptools.build_meta` |
| Author(s) | Erin Patrick Spencer <wayseer@interdependentway.org> |
| Repository | https://github.com/The-Interdependency/pcta |
| Runtime dependencies | none (stdlib only) |
| Optional extras | `dev` |
| Keywords | none |
| CI workflows | `ci.yml`, `manifest-check.yml` |
| Top-level directories | `pcta/` · `tests/` |

<sub>Derived from `pyproject.toml` + the repo tree. Unknown fields surface as `hmmm` rather than a guess.</sub>
<!-- END GENERATED:manifest -->
> The block above is generated from `pyproject.toml` + the repo tree by the
> `manifest` living-spec tool (`.agents/skills/manifest/`) and gated in CI
> (`.github/workflows/manifest-check.yml`) — do not hand-edit between the
> markers; run `python .agents/skills/manifest/generate.py --write` after
> changing version/deps/layout.

**License:** AGPL-3.0-or-later, dual-licensed with a commercial option (see
`LICENSE` / `LICENSE-COMMERCIAL.md`). Copyright (c) 2026 Erin Patrick Spencer.

---

## Place in the prime-tensor stack

The canonical role-and-boundary map is the **single source of truth**:
`The-Interdependency/interdependent-lib : docs/prime-tensor-stack.md`. This repo
**cites** it and never imports it; cross-repo interoperability is **not**
continuity and moves no theorem / proof / empirical status between repos.

```
PCNA (tensors → circles, backprop) ─► circles ─► PCTA (circles → seeds)
  ─► seeds ─► PTCA (seeds → core) ─► cores ─► a0(zfae) inference
PCEA — guardian: last-state-as-key encryption at every layer (orthogonal)
```

Composition counts are **variable** at every level — the only invariant is that
every circle, seed, and core is itself a tensor.

| # | Layer | Repo | Expansion | Role |
|---|-------|------|-----------|------|
| 1 | Tensor / Circle | `pcna` | Prime Circle Neural Architecture | Arranges tensors as **circles** in a standard back-propagating NN (only differentiable layer); offers circles to PCTA → **weights** |
| **2** | **Seed** | **`pcta` (this repo)** | **Prime Circled Tensor Architecture** | **Composes UCNS-carried circles into seeds; offers seeds to PTCA → structural motion** |
| 3 | Core | `ptca` / `ptca-lib` | Prime Tensor Core Architecture | Composes seeds into **cores**; offers cores to `a0(zfae)` → structural motion |
| — | Inference | `zfae` (runtime in `a0`) | Zeta Function Alpha Echo | Uses pcna tensors as weights; pcna circles / pcta seeds / ptca cores as phase-harmonic propagation + auditing |
| — | Guardian | `pcea` | Prime Circular Encryption Algorithm | "Last state as key for this state" encryption at every layer — orthogonal, not a layer |

---

## Repository Layout

```
pcta/
  __init__.py     Public surface (re-exports below) + __version__
  constants.py    routing motif (HEPTAGRAM_VERTICES, SEED_ROUTING_STEP={7/3}), coherence-prime guard
  compose.py      heptagram_order, compose_seed (the ⊠ operator; variable count), build_seed, seed_motion
  tensor.py       CircleTensor (opaque UCNS-carried layer-1 tensor), Seed (produced tensor), SeedMotion

tests/            stdlib unittest suite (also runs under pytest)
  test_constants.py   composition counts + coherence-prime ladder (incl. p=4373 regression)
  test_compose.py     heptagram order, compose_seed, build_seed, seed_motion

.github/workflows/
  ci.yml              test gate (matrix 3.9/3.11/3.13; editable install + unittest discover)
  manifest-check.yml  living-spec drift gate (vendored generate.py checksum + --check)

.agents/skills/   vendored org skills (manifest, meta-module-build, msdmd, test-build)
pyproject.toml    setuptools; name=pcta; AGPL-3.0-or-later; py>=3.9; zero deps; dev=[pytest]
README.md  LICENSE  LICENSE-COMMERCIAL.md
```

---

## Development Workflow

```bash
# Editable install (zero runtime deps; add dev extra for pytest)
python -m pip install -e .
python -m pip install -e ".[dev]"

# Run the test suite (stdlib — no deps needed). This is what CI runs.
python -m unittest discover -s tests -v
# ... or under pytest (dev extra):
pytest

# Import smoke
python -c "import pcta; print(pcta.__version__)"

# Refresh the generated CLAUDE.md manifest block (after pyproject/layout changes)
python .agents/skills/manifest/generate.py --write
```

CI: `ci.yml` runs the unittest suite on Python 3.9/3.11/3.13; `manifest-check.yml`
verifies the vendored `generate.py` checksum and that the manifest block is in sync.

---

## Public Surface

`pcta/__init__.py` is the stable API.

| Group | Names |
|-------|-------|
| Objects | `CircleTensor`, `Seed`, `SeedMotion` |
| Composition | `compose_seed`, `build_seed`, `seed_motion`, `heptagram_order` |
| Constants / guard | `NOMINAL_CIRCLES_PER_SEED`, `HEPTAGRAM_VERTICES`, `SEED_ROUTING_STEP`, `is_coherence_prime`, `coherence_primes_up_to`, `nth_coherence_prime` |

- `compose_seed(circles)` — the structural `⊠` operator: assigns the
  `{n/3}` star-polygon anchor order (the `{7/3}` heptagram for the nominal
  `n=7`) and grafts a **variable** number of circles into a `Seed`. Falls back
  to identity order when `gcd(step, n) != 1`.
- `seed_motion(seed)` — the structural **motion** handed upward: identity +
  star-polygon order + circle identities. No weights, no gradients.

---

## Key Conventions & Gotchas

- **Variable composition.** A seed may carry any number of circles; the only
  invariant is that every circle is a tensor and the seed is itself a tensor.
  `NOMINAL_CIRCLES_PER_SEED`/`HEPTAGRAM_VERTICES` describe the routing motif, not
  a required count. Do not reintroduce a fixed circle count.
- **Structural / non-differentiable.** Composition creates no autodiff node;
  `requires_grad` is always `False`. Back-propagation lives **only** in `pcna`
  (layer 1). Do not add gradient logic here — that inverts the stack boundary.
- **Circles are opaque.** A circle's internal structure is `pcna`'s business.
  pcta carries the circle payload losslessly and never reads into it.
- **No runtime dependencies** — stdlib only. Keep it that way.
- **Coherence-prime rule is mirrored, not imported.** `constants.is_coherence_prime`
  reproduces `interdependent_lib.coherence_primes` verbatim (recursive ancestry,
  square-free kernel; includes the `p=4373` regression). Importing the aggregator
  would invert the dependency graph.
- **`hmmm` — do not encode as fact** (per org doctrine): the **formal definition
  of "motion"** (described by role only). The PCTA acronym expansion (Prime
  Circled Tensor Architecture) and the variable-count rule were resolved by the
  maintainer on 2026-06-05.
- **Manifest block is machine-owned.** Never hand-edit between the
  `BEGIN/END GENERATED:manifest` markers; run `generate.py --write`.

---

## Related Repos

| Repo | Role |
|------|------|
| The-Interdependency/interdependent-lib | Hosts the stack canon (`docs/prime-tensor-stack.md`) + `coherence_primes` |
| The-Interdependency/pcna | Layer 1 — tensors + back-propagation → weights |
| The-Interdependency/PTCA | Layer 3 — seeds → core (`ptca-lib`; `prime_core` experiment) |
| The-Interdependency/ZFAE | Inference cap — reads weights + motion (runtime in `a0`) |
| The-Interdependency/PCEA | Orthogonal guardian — seals weights / state |

---

## Git Workflow

- Main branch: `main`
- Feature branches: `feat/<desc>`, `fix/<desc>`, `docs/<desc>`, `chore/<desc>`
- Commit style: Conventional Commits (`feat(pcta):`, `fix(compose):`, etc.)
- Author: Erin Patrick Spencer (wayseer@interdependentway.org)
- License: AGPL-3.0-or-later (dual-licensed commercial option)

## Agent module-build doctrine

Before adding a new module, route, service, adapter, schema, worker, engine,
UI panel, migration, or experiment, read:

`./.agents/skills/meta-module-build/SKILL.md`

New module work should start with a `MODULE_BUILD` block (see the example in
`pcta/constants.py`). Unknown fields must be marked `hmmm`, not guessed.
