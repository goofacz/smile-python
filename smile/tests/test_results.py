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
        self.result.position_x = 10
        self.result.position_y = 20
        self.result.position_z = 30
        self.result.begin_true_position_x = 400
        self.result.begin_true_position_y = 500
        self.result.begin_true_position_z = 600
        self.result.end_true_position_x = 7000
        self.result.end_true_position_y = 8000
        self.result.end_true_position_z = 9000

    def test_to_list_success(self):
        reference_tuple = (
            112233445566,
            2,
            10,
            20,
            30,
            400,
            500,
            600,
            7000,
            8000,
            9000
        )

        self.assertSequenceEqual(self.result.to_tuple(), reference_tuple)

    def test_to_list_catch_nones(self):
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


if __name__ == '__main__':
    unittest.main()
