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

from smile.result import Result
import smile.results as sresults
from smile.visualization import *
import unittest
from math import sqrt


class TestAnalysis(unittest.TestCase):
    def setUp(self):
        self.dummy_result = Result()
        self.dummy_result.mac_address = 0
        self.dummy_result.position_dimensions = 2
        self.dummy_result.position_x = 0.
        self.dummy_result.position_y = 0.
        self.dummy_result.position_z = 0.
        self.dummy_result.begin_true_position_x = 0.
        self.dummy_result.begin_true_position_y = 0.
        self.dummy_result.begin_true_position_z = 0.
        self.dummy_result.end_true_position_x = 0.
        self.dummy_result.end_true_position_y = 0.
        self.dummy_result.end_true_position_z = 0.

    def test_absolute_position_error_histogram_multiple_elements(self):
        results = [self.dummy_result, self.dummy_result, self.dummy_result, self.dummy_result]
        results = sresults.create_results(results)
        results.loc[0, ["begin_true_position_x", "begin_true_position_y"]] = (0, 0)
        results.loc[0, ["position_x", 'position_y']] = (15, 0)
        results.loc[0, ["end_true_position_x", "end_true_position_y"]] = (20, 0)

        results.loc[1, ["begin_true_position_x", "begin_true_position_y"]] = (0, 0)
        results.loc[1, ["position_x", 'position_y']]= (0, 5)
        results.loc[1, ["end_true_position_x", "end_true_position_y"]] = (0, 40)

        results.loc[2, ["begin_true_position_x", "begin_true_position_y"]] = (0, 0)
        results.loc[2, ["position_x", 'position_y']] = (6, 6)
        results.loc[2, ["end_true_position_x", "end_true_position_y"]] = (20, 20)

        results.loc[3, ["begin_true_position_x", "begin_true_position_y"]] = (0, 0)
        results.loc[3, ["position_x", 'position_y']] = (np.nan, np.nan)
        results.loc[3, ["end_true_position_x", "end_true_position_y"]] = (30, 30)

        true_position, position_errors, nans_number = plot_absolute_position_error_histogram(results,
                                                                                return_intermediate_results=True)
        self.assertTupleEqual((4, 3), true_position.shape)
        np.testing.assert_equal((10, 0, 0), true_position[0, :])
        np.testing.assert_equal((0, 20, 0), true_position[1, :])
        np.testing.assert_equal((10, 10, 0), true_position[2, :])

        self.assertTupleEqual((3,), position_errors.shape)
        self.assertEqual(5, position_errors[0])
        self.assertEqual(15, position_errors[1])
        self.assertEqual(1, nans_number)
        self.assertAlmostEqual(sqrt(2 * 10 ** 2) - sqrt(2 * 6 ** 2), position_errors[2], places=10)


if __name__ == '__main__':
    unittest.main()
