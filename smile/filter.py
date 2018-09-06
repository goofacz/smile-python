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

import numpy as np


class Filter:
    """
    Filters out data from Array-based or numpy.ndarray arrays.

    >>> # Column names are: 'first':0, 'second':1, 'third'"2
    >>> data = TestArray.Data([[1, 10, 100],
    >>>                        [2, 20, 200],
    >>>                        [3, 30, 300]])
    >>>
    >>> filter = Filter(data)
    >>> filter.is_in('second', (10, 20))
    >>> filter.equal(2, 200)
    >>> result = filter.finish() # Result: [2, 20, 200]
    """
    def __init__(self, array):
        self._array = array

    def less_equal(self, column, value):
        """
        Removes rows for which value at specified column is not <= value.

        Args:
            column: (int, str) Numeric index or column name.
            value: The value.
        """
        self._array = self._array[np.less_equal(self._array[:, column], value)]
        return self

    def less(self, column, value):
        """
        Removes rows for which value at specified column is not < value.

        Args:
            column: (int, str) Numeric index or column name.
            value: The value.
        """
        self._array = self._array[np.less(self._array[:, column], value)]
        return self

    def greater_equal(self, column, value):
        """
        Removes rows for which value at specified column is not >= value.

        Args:
            column: (int, str) Numeric index or column name.
            value: The value.
        """
        self._array = self._array[np.greater_equal(self._array[:, column], value)]
        return self

    def greater(self, column, value):
        """
        Removes rows for which value at specified column is not > value.

        Args:
            column: (int, str) Numeric index or column name.
            value: The value.
        """
        self._array = self._array[np.greater(self._array[:, column], value)]
        return self

    def equal(self, column, value):
        """
        Removes rows for which value at specified column is not == value.

        Args:
            column: (int, str) Numeric index or column name.
            value: The value.
        """
        self._array = self._array[np.equal(self._array[:, column], value)]
        return self

    def not_equal(self, column, value):
        """
        Removes rows for which value at specified column in not != value.

        Args:
            column: (int, str) Numeric index or column name.
            value: The value.
        """
        self._array = self._array[np.not_equal(self._array[:, column], value)]
        return self

    def is_in(self, column, values):
        """
        Removes rows for which value at specified column is in the set of values.

        Args:
            column: (int, str) Numeric index or column name.
            values: The set of values.
        """
        self._array = self._array[np.isin(self._array[:, column], values)]
        return self

    def is_not_in(self, column, values):
        """
        Removes rows for which value at specified column is out of the set of values.

        Args:
            column: (int, str) Numeric index or column name.
            values: The set of values.
        """
        self._array = self._array[np.isin(self._array[:, column], values, invert=True)]
        return self

    def finish(self):
        """
        Finishes filtering process.

        Returns:
            Array with filtered out data.
        """
        return self._array
