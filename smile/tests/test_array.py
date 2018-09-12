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

from smile.array import *


class Data(Array):
    def __new__(cls, input_array):
        column_names = {
            'first': 0,
            'second': 1,
            'third': 2
        }

        return super(Data, cls).__new__(cls, input_array, column_names, None)


class TestArrayConstruct(unittest.TestCase):
    def test_construction(self):
        Data([0, 1, 2])


class TestArrayIndexing(unittest.TestCase):
    def setUp(self):
        self.data = Data([[0, 1, 2],
                          [3, 4, 5],
                          [6, 7, 8]])

    # Single element
    def test_single_element_by_int(self):
        result = self.data[1, 2]
        self.assertTrue(isinstance(result, np.int64))
        self.assertEqual(result, 5)

        self.data[2, 1] = -1
        np.testing.assert_equal(self.data, [[0, 1, 2],
                                            [3, 4, 5],
                                            [6, -1, 8]])

    def test_single_element_by_int_str(self):
        result = self.data[1, 'first']
        self.assertTrue(isinstance(result, np.int64))
        self.assertEqual(result, 3)

        self.data[2, 'first'] = 6
        np.testing.assert_equal(self.data, [[0, 1, 2],
                                            [3, 4, 5],
                                            [6, 7, 8]])

    # Single row
    def test_single_row_by_int(self):
        result = self.data[2]
        self.assertTrue(isinstance(result, Data))
        np.testing.assert_equal(result, [6, 7, 8])

        self.data[2] = [-1, -2, -3]
        np.testing.assert_equal(self.data, [[0, 1, 2],
                                            [3, 4, 5],
                                            [-1, -2, -3]])

    def test_single_row_by_int_range(self):
        result = self.data[2, :]
        self.assertTrue(isinstance(result, Data))
        np.testing.assert_equal(result, [6, 7, 8])

        self.data[2, :] = [-1, -2, -3]
        np.testing.assert_equal(self.data, [[0, 1, 2],
                                            [3, 4, 5],
                                            [-1, -2, -3]])

    def test_single_row_by_int_ints(self):
        result = self.data[2, [0, 1, 2]]
        self.assertTrue(isinstance(result, Data))
        np.testing.assert_equal(result, [6, 7, 8])

        self.data[1, [0, 1, 2]] = [-1, -2, -3]
        np.testing.assert_equal(self.data, [[0, 1, 2],
                                            [-1, -2, -3],
                                            [6, 7, 8]])

    def test_single_row_by_int_strs(self):
        result = self.data[1, ['first', 'second', 'third']]
        self.assertTrue(isinstance(result, Data))
        np.testing.assert_equal(result, [3, 4, 5])

        self.data[0, ['first', 'second', 'third']] = [-1, -2, -3]
        np.testing.assert_equal(self.data, [[-1, -2, -3],
                                            [3, 4, 5],
                                            [6, 7, 8]])

    def test_single_row_by_int_mixed(self):
        result = self.data[1, ['first', 'second', 2]]
        self.assertTrue(isinstance(result, Data))
        np.testing.assert_equal(result, [3, 4, 5])

        self.data[1, ['first', 'second', 2]] = [-1, -2, -3]
        np.testing.assert_equal(self.data, [[0, 1, 2],
                                            [-1, -2, -3],
                                            [6, 7, 8]])

    # Single column
    def test_single_column_by_str(self):
        result = self.data[:, 'second']
        self.assertTrue(isinstance(result, Data))
        np.testing.assert_equal(result, [1, 4, 7])

        self.data[:, 'first'] = [-1, -2, -3]
        np.testing.assert_equal(self.data, [[-1, 1, 2],
                                            [-2, 4, 5],
                                            [-3, 7, 8]])

    def test_single_column_by_range_int(self):
        result = self.data[:, 0]
        self.assertTrue(isinstance(result, Data))
        np.testing.assert_equal(result, [0, 3, 6])

        self.data[:, 1] = [-1, -2, -3]
        np.testing.assert_equal(self.data, [[0, -1, 2],
                                            [3, -2, 5],
                                            [6, -3, 8]])

    def test_single_column_by_range_str(self):
        result = self.data[:, 'second']
        self.assertTrue(isinstance(result, Data))
        np.testing.assert_equal(result, [1, 4, 7])

        self.data[:, 'second'] = [-1, -2, -3]
        np.testing.assert_equal(self.data, [[0, -1, 2],
                                            [3, -2, 5],
                                            [6, -3, 8]])

    def test_single_column_by_ints_int(self):
        result = self.data[[0, 1, 2], 0]
        self.assertTrue(isinstance(result, Data))
        np.testing.assert_equal(result, [0, 3, 6])

        self.data[[0, 1, 2], 0] = [-1, -2, -3]
        np.testing.assert_equal(self.data, [[-1, 1, 2],
                                            [-2, 4, 5],
                                            [-3, 7, 8]])

    def test_single_column_by_ints_str(self):
        result = self.data[[0, 1, 2], 'second']
        self.assertTrue(isinstance(result, Data))
        np.testing.assert_equal(result, [1, 4, 7])

        self.data[[0, 1, 2], 'third'] = [-1, -2, -3]
        np.testing.assert_equal(self.data, [[0, 1, -1],
                                            [3, 4, -2],
                                            [6, 7, -3]])

    # Subset
    def test_subset_1(self):
        result = self.data[1:, 'second']
        self.assertTrue(isinstance(result, Data))
        np.testing.assert_equal(result, [4, 7])

        self.data[:2, ['second', 'first']] = [[-1, -2],
                                              [-3, -4]]

        np.testing.assert_equal(self.data, [[-2, -1, 2],
                                            [-4, -3, 5],
                                            [6, 7, 8]])

    def test_subset_2(self):
        result = self.data[[0, 2], [0, 'third']]
        self.assertTrue(isinstance(result, Data))
        np.testing.assert_equal(result, [0, 8])

        self.data[[0, 2], [0, 'third']] = [-1, -2]
        np.testing.assert_equal(self.data, [[-1, 1, 2],
                                            [3, 4, 5],
                                            [6, 7, -2]])

    # Invalid indexing
    def test_invalid_row_by_str(self):
        with self.assertRaises(IndexError):
            self.data['one']

    def test_invalid_row_by_str_int(self):
        with self.assertRaises(IndexError):
            self.data['one', 0]

    def test_invalid_row_by_str_range(self):
        with self.assertRaises(IndexError):
            self.data['one', :]

    def test_invalid_column_by_incorrect_str(self):
        with self.assertRaises(Array.UnknownColumnLabelError):
            self.data[0, 'tenth']


class TestArrayFindOrder(unittest.TestCase):
    def setUp(self):
        self.data = Data([[1, 2, 3],
                          [7, 8, 9],
                          [4, 5, 6]])

    def test_success(self):
        indices = self.data.first.find_order(np.asarray([4, 1, 7]))
        result = self.data[indices, :]
        np.testing.assert_equal(result, [[4, 5, 6],
                                         [1, 2, 3],
                                         [7, 8, 9]])

    def test_self_is_array(self):
        with self.assertRaises(ValueError):
            self.data.find_order(np.asarray([4, 1, 1]))

    def test_values_is_array(self):
        with self.assertRaises(ValueError):
            self.data.first.find_order(np.asarray([[4, 1, 7],
                                                   [5, 9, 2]]))

    def test_values_array_diff_size(self):
        with self.assertRaises(ValueError):
            self.data.first.find_order(np.asarray([4, 1, 7, 9]))

    def test_values_not_in_array(self):
        with self.assertRaises(ValueError):
            self.data.first.find_order(np.asarray([4, 1, 9]))

    def test_non_unique_values(self):
        with self.assertRaises(Array.NonUniqueValuesError):
            self.data.first.find_order(np.asarray([4, 1, 1]))

    def test_non_unique_column(self):
        invalid_data = Data([[1, 2, 3],
                             [1, 8, 9],
                             [4, 5, 6]])

        with self.assertRaises(Array.NonUniqueValuesError):
            invalid_data.first.find_order(np.asarray([4, 1, 7]))


class TestSort(unittest.TestCase):
    def test_sort(self):
        data = np.array([[4, 5],
                         [8, 2],
                         [1, 8]])

        np.testing.assert_equal(sort(data, 0), [[1, 8],
                                                [4, 5],
                                                [8, 2]])


if __name__ == '__main__':
    unittest.main()
