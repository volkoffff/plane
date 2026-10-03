from __future__ import annotations

import unittest

import numpy as np

from physics.collisions import segment_box_entry


class SegmentBoxTests(unittest.TestCase):
    def test_crossing_in_both_directions_and_parallel_axes(self) -> None:
        with np.errstate(all="raise"):
            for start, end in (((-3, 0, 0), (3, 0, 0)), ((3, 0, 0), (-3, 0, 0))):
                self.assertAlmostEqual(
                    segment_box_entry(np.array(start), np.array(end), (1, 1, 1)),
                    1 / 3,
                )

    def test_parallel_segment_outside_box(self) -> None:
        self.assertIsNone(
            segment_box_entry(np.array([-3, 2, 0]), np.array([3, 2, 0]), (1, 1, 1))
        )

    def test_start_inside_and_stationary_point(self) -> None:
        for end in ((0, 0, 0), (3, 0, 0)):
            self.assertEqual(
                segment_box_entry(np.zeros(3), np.array(end), (1, 1, 1)), 0.0
            )
        self.assertIsNone(
            segment_box_entry(np.array([2, 0, 0]), np.array([2, 0, 0]), (1, 1, 1))
        )

    def test_contact_with_face_and_endpoint_counts_as_hit(self) -> None:
        self.assertAlmostEqual(
            segment_box_entry(np.array([-3, 1, 0]), np.array([-1, 1, 0]), (1, 1, 1)),
            1.0,
        )

    def test_line_hits_but_segment_does_not(self) -> None:
        self.assertIsNone(
            segment_box_entry(np.array([-3, 0, 0]), np.array([-2, 0, 0]), (1, 1, 1))
        )
        self.assertIsNone(
            segment_box_entry(np.array([2, 0, 0]), np.array([3, 0, 0]), (1, 1, 1))
        )

    def test_invalid_geometry_is_rejected(self) -> None:
        for start, end, extents in (
            ([0, 0], [1, 0, 0], (1, 1, 1)),
            ([0, 0, 0], [1, float("nan"), 0], (1, 1, 1)),
            ([0, 0, 0], [1, 0, 0], (0, 1, 1)),
        ):
            with self.assertRaises(ValueError):
                segment_box_entry(np.array(start), np.array(end), extents)


if __name__ == "__main__":
    unittest.main()
