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

# There's no need to display interactive plots
import matplotlib

matplotlib.use('AGG')

import unittest
import numpy.testing as npt
from smile.result import Result
import smile.results as sresults
import smile.analysis as sanalysis


class TestAnalysis(unittest.TestCase):
    def test_compute_basic_statistics(self):
        result1 = Result()
        result1.mac_address = 112233445566
        result1.position_dimensions = 2
        result1.position_x = 1
        result1.position_y = 2
        result1.position_z = 3
        result1.begin_true_position_x = 4
        result1.begin_true_position_y = 5
        result1.begin_true_position_z = 6
        result1.end_true_position_x = 5
        result1.end_true_position_y = 6
        result1.end_true_position_z = 7
        result1.reference_position_x = 4.5
        result1.reference_position_y = 5.5
        result1.reference_position_z = 6.5

        result2 = Result()
        result2.mac_address = 112233445566
        result2.position_dimensions = 2
        result2.position_x = 2
        result2.position_y = 4
        result2.position_z = 6
        result2.begin_true_position_x = 4
        result2.begin_true_position_y = 5
        result2.begin_true_position_z = 6
        result2.end_true_position_x = 5
        result2.end_true_position_y = 6
        result2.end_true_position_z = 7
        result2.reference_position_x = 4.5
        result2.reference_position_y = 5.5
        result2.reference_position_z = 6.5

        results = sresults.create_results([result1, result2])
        statistics = sanalysis.compute_basic_statistics(results)

        self.assertEqual(112233445566, statistics['mac_address'][0])
        self.assertEqual(4.5, statistics['reference_position_x'][0])
        self.assertEqual(5.5, statistics['reference_position_y'][0])
        self.assertEqual(6.5, statistics['reference_position_z'][0])

        # positions
        self.assertEqual(2, statistics['positions']['count'][0])

        # position_x
        self.assertEqual(1, statistics['position_x']['min'][0])
        self.assertEqual(2, statistics['position_x']['max'][0])
        npt.assert_array_almost_equal(0.707107, statistics['position_x']['std'][0])
        self.assertEqual(0.5, statistics['position_x']['var'][0])
        self.assertEqual(1.25, statistics['position_x']['25%'][0])
        self.assertEqual(1.5, statistics['position_x']['50%'][0])
        self.assertEqual(1.75, statistics['position_x']['75%'][0])

        # position_y
        self.assertEqual(2, statistics['position_y']['min'][0])
        self.assertEqual(4, statistics['position_y']['max'][0])
        npt.assert_array_almost_equal(1.414214, statistics['position_y']['std'][0])
        self.assertEqual(2, statistics['position_y']['var'][0])
        self.assertEqual(2.5, statistics['position_y']['25%'][0])
        self.assertEqual(3, statistics['position_y']['50%'][0])
        self.assertEqual(3.5, statistics['position_y']['75%'][0])

        # position_y
        self.assertEqual(3, statistics['position_z']['min'][0])
        self.assertEqual(6, statistics['position_z']['max'][0])
        npt.assert_array_almost_equal(2.12132, statistics['position_z']['std'][0])
        self.assertEqual(4.5, statistics['position_z']['var'][0])
        self.assertEqual(3.75, statistics['position_z']['25%'][0])
        self.assertEqual(4.5, statistics['position_z']['50%'][0])
        self.assertEqual(5.25, statistics['position_z']['75%'][0])


if __name__ == '__main__':
    unittest.main()
