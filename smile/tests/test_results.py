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

from smile.result import Result
import smile.results as sresults


class TestResults(unittest.TestCase):
    def setUp(self):
        self.result1 = Result()
        self.result1.mac_address = 112233445566
        self.result1.position_dimensions = 2
        self.result1.position_x = 1
        self.result1.position_y = 2
        self.result1.position_z = 3
        self.result1.begin_true_position_x = 0
        self.result1.begin_true_position_y = 0
        self.result1.begin_true_position_z = 0
        self.result1.end_true_position_x = 2
        self.result1.end_true_position_y = 2
        self.result1.end_true_position_z = 2
        self.result1.reference_position_x = 1
        self.result1.reference_position_y = 1
        self.result1.reference_position_z = 1

        self.result2 = Result()
        self.result2.mac_address = 212233445566
        self.result2.position_dimensions = 2
        self.result2.position_x = 10
        self.result2.position_y = 20
        self.result2.position_z = 30
        self.result2.begin_true_position_x = 10
        self.result2.begin_true_position_y = 10
        self.result2.begin_true_position_z = 0
        self.result2.end_true_position_x = 20
        self.result2.end_true_position_y = 20
        self.result2.end_true_position_z = 0
        self.result2.reference_position_x = 15
        self.result2.reference_position_y = 15
        self.result2.reference_position_z = 0

        self.result3 = Result()
        self.result3.mac_address = 312233445566
        self.result3.position_dimensions = 2
        self.result3.position_x = 100
        self.result3.position_y = 200
        self.result3.position_z = 300
        self.result3.begin_true_position_x = 100
        self.result3.begin_true_position_y = 100
        self.result3.begin_true_position_z = 0
        self.result3.end_true_position_x = 100
        self.result3.end_true_position_y = 100
        self.result3.end_true_position_z = 100
        self.result3.reference_position_x = 100
        self.result3.reference_position_y = 100
        self.result3.reference_position_z = 50

        self.reference_results = [self.result1, self.result2, self.result3]

    def test_create_array_success(self):
        results = sresults.create_results(self.reference_results)

        self.assertEqual(results.shape, (3, 14))
        np.testing.assert_array_equal(results.iloc[0, :], self.result1.to_tuple())
        np.testing.assert_array_equal(results.iloc[1, :], self.result2.to_tuple())
        np.testing.assert_array_equal(results.iloc[2, :], self.result3.to_tuple())

    def test_create_array_different_position_dimensions(self):
        self.result1.position_dimensions = 2
        self.result2.position_dimensions = 3
        self.result3.position_dimensions = 2

        self.reference_results = [self.result1, self.result2, self.result3]

        with self.assertRaises(ValueError):
            sresults.create_results(self.reference_results)

    def test_create_array_invalid_position_dimensions(self):
        self.result1.position_dimensions = -1
        self.result2.position_dimensions = 4
        self.result3.position_dimensions = 5

        with self.assertRaises(ValueError):
            sresults.create_results([self.result1])
        with self.assertRaises(ValueError):
            sresults.create_results([self.result2])
        with self.assertRaises(ValueError):
            sresults.create_results([self.result3])