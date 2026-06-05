"""Tests for pcta.compose / pcta.tensor — the circles -> seed operator."""
from __future__ import annotations

import unittest

from pcta import (
    CIRCLES_PER_SEED,
    CircleTensor,
    Seed,
    SeedMotion,
    build_seed,
    compose_seed,
    heptagram_order,
    seed_motion,
)


class TestHeptagramOrder(unittest.TestCase):
    def test_known_orders(self):
        self.assertEqual(heptagram_order(2), [0, 2, 4, 6, 1, 3, 5])
        self.assertEqual(heptagram_order(3), [0, 3, 6, 2, 5, 1, 4])

    def test_single_cycle_visits_every_vertex(self):
        self.assertEqual(sorted(heptagram_order(3)), list(range(7)))

    def test_non_coprime_step_rejected(self):
        # gcd(7, 7) == 7 -> not a single cycle.
        with self.assertRaises(ValueError):
            heptagram_order(7, 7)


class TestComposeSeed(unittest.TestCase):
    def _seven_circles(self):
        return [CircleTensor(payload=f"w{i}", identity=f"c{i}") for i in range(7)]

    def test_strict_requires_seven(self):
        with self.assertRaises(ValueError):
            compose_seed(self._seven_circles()[:3])  # too few, strict default

    def test_compose_assigns_heptagram_anchors(self):
        seed = compose_seed(self._seven_circles())
        self.assertIsInstance(seed, Seed)
        self.assertEqual(seed.n_circles, CIRCLES_PER_SEED)
        self.assertEqual(seed.anchor_order, tuple(heptagram_order(3)))
        # input circle i lands at heptagram position order[i]
        order = heptagram_order(3)
        for i in range(7):
            self.assertEqual(seed.at(order[i]).identity, f"c{i}")

    def test_lossless_payload_roundtrip(self):
        seed = compose_seed(self._seven_circles())
        payloads = {c.identity: c.payload for c in seed.circles}
        self.assertEqual(payloads["c0"], "w0")
        self.assertEqual(payloads["c6"], "w6")

    def test_non_differentiable(self):
        seed = compose_seed(self._seven_circles())
        self.assertFalse(seed.requires_grad)
        self.assertFalse(seed.circles[0].requires_grad)

    def test_partial_seed_allowed(self):
        seed = compose_seed(self._seven_circles()[:3], strict=False)
        self.assertEqual(seed.n_circles, 3)


class TestBuildSeedAndMotion(unittest.TestCase):
    def test_build_seed_from_payloads(self):
        seed = build_seed([f"weights{i}" for i in range(7)], identity="seed:13")
        self.assertEqual(seed.identity, "seed:13")
        self.assertEqual(seed.n_circles, 7)
        self.assertEqual(seed.at(0).identity, "seed:13.c0")

    def test_seed_motion_is_structural_only(self):
        seed = build_seed([f"w{i}" for i in range(7)], identity="seed:13")
        motion = seed_motion(seed)
        self.assertIsInstance(motion, SeedMotion)
        self.assertEqual(motion.seed_identity, "seed:13")
        self.assertEqual(motion.routing_step, 3)
        self.assertEqual(motion.anchor_order, tuple(heptagram_order(3)))
        # circle identities are reported in heptagram order, carrying no weights
        self.assertEqual(len(motion.circle_identities), 7)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
