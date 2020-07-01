import unittest
from typing import List, Tuple

from src import polygon_tools as pt


class Index2PointTests(unittest.TestCase):

    def test_simple(self):
        indices: Tuple[int, ...] = (83, 84, 85)
        points: List[Tuple[int, int]] = [(2, 27), (3, 0), (3, 1)]
        for idx, p in zip(indices, points):
            point: Tuple[int, int] = pt.index2point(idx)
            self.assertEqual(point, p)


class EuclideanDistanceTests(unittest.TestCase):

    def test_simple(self):
        distance: float = pt.euclidean_distance(0, 0, 1, 1)
        self.assertAlmostEqual(distance, 1.41421356)


class SortPolygonTests(unittest.TestCase):

    def test_simple(self):
        polygon: List[Tuple[int, int]] = [(0, 0), (5, 5), (1, 1)]
        p_sorted: List[Tuple[int, int]] = pt.sort_polygon(polygon)

        self.assertEqual(p_sorted, [(1, 1), (0, 0), (5, 5)])


class GetSizeTests(unittest.TestCase):

    def test_simple(self):
        polygon: List[List[int]] = [
            [0, 10],
            [5, 8]
        ]
        x_min, x_max, y_min, y_max, height, width = pt.get_size(polygon)

        self.assertEqual(x_min, polygon[0][0])
        self.assertEqual(x_max, polygon[1][0])
        self.assertEqual(y_min, polygon[1][1])
        self.assertEqual(y_max, polygon[0][1])
        self.assertEqual(height, polygon[0][1] - polygon[1][1])
        self.assertEqual(width, polygon[1][0] - polygon[0][0])
