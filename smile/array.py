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


class Array(np.ndarray):
    """
    Array extends NumPy ndarray with following features:

    - Human-readable column names.
    - Enriched indexing using column labels in brackets and as attributes.

    Array remains fully compatible wth NumPy routines. It is intended to be
    used as a base class for other array types.

    Array uses float64 as base type.

    Example showing how to inherit from Array and how to use new indexing
    capabilities.

    >>> class Data(Array):
    >>>     def __new__(cls, input_array):
    >>>         column_names = {
    >>>             'first': 0,
    >>>             'second': 1,
    >>>             'third': 2,
    >>>         }
    >>>         return super(Data, cls).__new__(cls, input_array)
    >>>
    >>> data = Data([[1, 10, 100],
    >>>              [2, 20, 200],
    >>>              [3, 30, 300]])
    >>>
    >>> result = data[(0, 2)] # Result: 100
    >>> result = data['third'] # Result: [100, 200, 300]
    >>> result = data[2, 'second'] # Result: 30
    >>> result = data[:, 'first'] # Result: [1, 2, 3]
    >>>
    >>> result = data.first # Result: [1, 2, 3]
    >>> result = data.first[1] # Result: 2
    """

    class InvalidColumnNameError(RuntimeError):
        """
        Raised when any of column names shadows any of ndarray's attribute name.
        """
        pass

    class InvalidArgumentError(RuntimeError):
        """
        Raised when Array is constructed with unacceptable input_array type.
        """
        pass

    class NonUniqueValuesError(RuntimeError):
        """
        Raised when elements in sequence are not unique.
        """
        pass

    def __new__(cls, input_array, column_names, column_converters=None):
        """
        Constructs new Array-based class instance.

        Args:
            input_array: (str, file-like, sequence) CSV file path, file, file-like object (e.g. StringIO) or
                                                    sequence of input data (e.g. list of lists).
            column_names: (dict) Column labels (label: index mapping).
            column_converters: (dict, optional) Functions converting columns' content to array's dtype.

        Returns:
            New class instance.
        """
        if isinstance(input_array, np.ndarray):
            pass
        elif isinstance(input_array, str) or hasattr(input_array, 'read'):
            # Catch strings (file paths) and file-like objects (file, StringIO)
            input_array = np.loadtxt(input_array, delimiter=',', converters=column_converters, ndmin=2,
                                     dtype=np.float64)
        elif hasattr(input_array, '__iter__'):
            # Catch sequences like lists and tuples
            input_array = np.asarray(input_array)
        else:
            raise Array.InvalidArgumentError(f'input_array has unacceptable type {str(type(input_array))}')

        instance = np.asarray(input_array).view(cls)
        instance.column_names = {}

        for column_name in column_names.keys():
            if hasattr(instance, column_name.lower()) or hasattr(instance, column_name.upper()):
                raise Array.InvalidColumnNameError(f'"{column_name}" cannot be used as column name, '
                                                   f'it shadows some ndarray attribute!')
        instance.column_names = column_names

        return instance

    def __array_finalize__(self, instance):
        if instance is None:
            return

        self.column_names = getattr(instance, 'column_names', None)

    def __array_wrap__(self, out_arr, context=None):
        return np.ndarray.__array_wrap__(self.view(type(self)), out_arr, context)

    def __getitem__(self, index):
        if len(self.shape) == 1:
            index = self._process_vector_index(index)
        else:
            index = self._process_array_index(index)

        return super(Array, self).__getitem__(index)

    def __setitem__(self, index, value):
        if len(self.shape) == 1:
            index = self._process_vector_index(index)
        else:
            index = self._process_array_index(index)

        return super(Array, self).__setitem__(index, value)

    def __getattribute__(self, item):
        if item == 'column_names' or item.startswith('_'):
            return super(Array, self).__getattribute__(item)
        else:
            try:
                return self[:, self.column_names[item]]
            except KeyError:
                return super(Array, self).__getattribute__(item)

    def _process_column_index(self, index):
        if isinstance(index, (int, slice, np.ndarray)):
            pass

        elif isinstance(index, str):
            try:
                index = self.column_names[index]
            except KeyError as error:
                raise IndexError(f'Unknown column name: {error}')

        elif isinstance(index, list):
            result = []
            for element in index:
                if isinstance(element, str):
                    try:
                        result.append(self.column_names[element])
                    except KeyError:
                        raise IndexError(f'Unknown column name: {element}')
                elif isinstance(element, int):
                    result.append(element)
                else:
                    raise IndexError(f'Invalid type of index element: {type(element)}')

            index = result

        else:
            raise IndexError(f'Invalid type of index element: {type(index)}')

        return index

    def _process_vector_index(self, index):
        return self._process_column_index(index)

    def _process_array_index(self, index):
        if isinstance(index, str):
            index = slice(None, None, None), self._process_column_index(index)

        elif isinstance(index, np.ndarray):
            pass

        elif isinstance(index, list):
            index = slice(None, None, None), self._process_column_index(index)

        elif isinstance(index, tuple):
            if len(index) == 1:
                index = index[0]
            else:
                index = index[0], self._process_column_index(index[1])

        else:
            raise IndexError(f'Invalid Array index: {index}')

        return index

    def find_order(self, target_values):
        """
        Finds indices of elements reordered by values from `target_values`.

        This method can be called only on vectors (columns or rows).

        Args:
            target_values (np.ndarray) : Desired order of elements. `target_values`
                                         has to be a vector.

        Returns:
            Indices that can be used to reorder array.

        Example:
            >>> data = Data([[1, 2, 3],
            >>>              [7, 8, 9],
            >>>              [4, 5, 6]])
            >>>
            >>> indices = self.data.[:,0].find_order(np.asarray([4, 1, 7]))
            >>> data = self.data[indices, :]
            >>> # data will contain
            >>> # [[4, 5, 6],
            >>> #  [1, 2, 3],
            >>> #  [7, 8, 9]])
        """
        if not isinstance(target_values, np.ndarray):
            raise ValueError('Input values has to be a np.ndarray')

        if (self.ndim != 1) or (target_values.ndim != 1):
            raise ValueError(f'Input values (ndim: {target_values.ndim}) and self '
                             f'(ndim: {self.ndim}) have to be a vectors')

        if len(self) != len(target_values):
            raise ValueError(f'Input values (len: {len(target_values)}) and self '
                             f'(len: {len(self)}) have to be the same size')

        _, counts = np.unique(target_values, return_counts=True)
        if any([count != 1 for count in counts]):
            raise Array.NonUniqueValuesError('Input values has to be unique')

        _, counts = np.unique(self, return_counts=True)
        if any([count != 1 for count in counts]):
            raise Array.NonUniqueValuesError('Column/row has to contain unique values')

        target_values = target_values.tolist()
        pairs = [(value, index[0]) for index, value in np.ndenumerate(self)]
        sorted_pairs = sorted(pairs, key=lambda pair: target_values.index(pair[0]))
        _, indices = zip(*sorted_pairs)

        return indices


def sort(array, column):
    return array[np.argsort(array[:, column]), :]
