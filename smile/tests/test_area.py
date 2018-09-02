#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see http:#www.gnu.org/licenses/.
#

import unittest

import numpy as np
import shapely.geometry as sg

import smile.area as sa


class TestArea(unittest.TestCase):
    def test_create_with_invalid_json(self):
        with self.assertRaises(sa.Area.InvalidContentError):
            sa.Area('resource:tests/area/invalid.json')

    def test_create_with_correct_json(self):
        area = sa.Area('resource:tests/area/square.json')
        self.assertIsInstance(area, sa.Area)


class TestAreaContains(unittest.TestCase):
    def setUp(self):
        self.area = sa.Area('resource:tests/area/square.json')

    def test_point_inside_area(self):
        inside_point = sg.Point(10, 10)
        self.assertTrue(self.area.contains(inside_point))

    def test_point_as_ndarray_inside_area(self):
        inside_point = np.asarray((10, 10))
        self.assertTrue(self.area.contains(inside_point))

    def test_point_as_tuple_inside_area(self):
        inside_point = (10, 10)
        self.assertTrue(self.area.contains(inside_point))

    def test_point_close_to_area(self):
        close_to_edge_point = sg.Point(-1e-6, 5)
        self.assertTrue(self.area.contains(close_to_edge_point))
        self.assertFalse(self.area.contains(close_to_edge_point, atol=1e-7))

    def test_point_outside_area(self):
        outside_point = sg.Point(100, 100)
        self.assertFalse(self.area.contains(outside_point))


if __name__ == '__main__':
    unittest.main()
