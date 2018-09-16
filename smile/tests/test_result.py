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
from copy import deepcopy
from smile.result import Result


class TestResult(unittest.TestCase):
    def setUp(self):
        self.result = Result()
        self.result.mac_address = 112233445566
        self.result.position_dimensions = 2
        self.result.position_x = 1
        self.result.position_y = 2
        self.result.position_z = 3
        self.result.begin_true_position_x = 4
        self.result.begin_true_position_y = 5
        self.result.begin_true_position_z = 6
        self.result.end_true_position_x = 5
        self.result.end_true_position_y = 6
        self.result.end_true_position_z = 7
        self.result.reference_position_x = 4.5
        self.result.reference_position_y = 5.5
        self.result.reference_position_z = 6.5

    def test_to_list_success(self):
        reference_result = (112233445566, 2, 1, 2, 3, 4, 5, 6, 5, 6, 7, 4.5, 5.5, 6.5)
        self.assertSequenceEqual(self.result.to_tuple(), reference_result)

    def test_to_list_none_values(self):
        none_result = Result()
        with self.assertRaises(ValueError):
            none_result.to_tuple()

        none_result = deepcopy(self.result)
        none_result.position_z = None
        with self.assertRaises(ValueError):
            none_result.to_tuple()

        none_result = deepcopy(self.result)
        none_result.begin_true_position_z = None
        with self.assertRaises(ValueError):
            none_result.to_tuple()

    def test_to_list_invalid_position_dimensions(self):
        self.result.position_dimensions = 0
        with self.assertRaises(ValueError):
            self.result.to_tuple()

        self.result.position_dimensions = 4
        with self.assertRaises(ValueError):
            self.result.to_tuple()


if __name__ == '__main__':
    unittest.main()

if __name__ == '__main__':
    unittest.main()
