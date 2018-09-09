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
    A thin wrapper for numpy.ndarray extending it's indexing capabilities. Array lets user
    to index arrays using human-readable column names inside brackets. Moreover, columns
    are accessible using attributes. Array remains full compatible wth umPy routines.

    Array is intended to be used as a base class and classes built on top of it
    defines human-readable column names.

    >>> class Data(Array):
    >>>     def __init__(self, *args):
    >>>         super(self.__class__, self).__init__()
    >>>         self.column_names['first'] = 0
    >>>         self.column_names['second'] = 1
    >>>         self.column_names['third'] = 2
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
        pass

    def __new__(cls, input_array, column_names, column_converters):
        if isinstance(input_array, np.ndarray):
            pass
        elif isinstance(input_array, str) or hasattr(input_array, 'read'):
            # Catch strings (file paths) and file-like objects (file, StringIO)
            input_array = np.loadtxt(input_array, delimiter=',', converters=column_converters, ndmin=2)
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
        index = self._process_index(index)
        return super(Array, self).__getitem__(index)

    def __setitem__(self, index, value):
        index = self._process_index(index)
        return super(Array, self).__setitem__(index, value)

    def __getattribute__(self, item):
        if item == 'column_names' or item.startswith('_'):
            return super(Array, self).__getattribute__(item)
        else:
            if item in self.column_names:
                return self[:, self.column_names[item]]
            else:
                return super(Array, self).__getattribute__(item)

    def _process_vector_index(self, index):
        if isinstance(index, str):
            if index not in self.column_names:
                raise IndexError(f'Unknown column name: {index}')

            return self.column_names[index]

        return index

    def _process_array_index(self, index):
        if isinstance(index, str):
            if index not in self.column_names:
                raise IndexError(f'Unknown column name: {index}')

            return slice(None, None, None), self.column_names[index]

        elif isinstance(index, (list, tuple)):
            if len(index) == 1:
                index = index[0]

            elif len(index) == 2:
                if isinstance(index[0], str):
                    raise IndexError('Rows cannot be indexed with string names')

                elif isinstance(index[0], (list, tuple)) and isinstance(index[0][0], np.ndarray):
                    index = (index[0][0], index[1])

                if isinstance(index[1], str):
                    if index[1] not in self.column_names:
                        raise IndexError(f'Unknown column name: {index[1]}')

                    index = (index[0], self.column_names[index[1]])

            else:
                raise IndexError(f'Sequence of {index} cannot be used for indexing')

        return index

    def _process_index(self, index):
        if len(self.shape) == 1:
            return self._process_vector_index(index)

        return self._process_array_index(index)


def sort(array, column):
    return array[np.argsort(array[:, column]), :]
