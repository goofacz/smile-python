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
    A thin wrapper for numpy.ndarray adjusting it to SMILe needs, while remaining
    fully compatible with numpy routines.

    Array lets user to index array using human-readable column names.

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
    """
    def __new__(cls, *args, **kargs):
        instance = np.asarray(*args, **kargs).view(cls)
        instance.column_names = {}
        return instance

    def __array_finalize__(self, instance):
        if instance is not None:
            self.column_names = getattr(instance, 'column_names', None)

    def __array_wrap__(self, out_arr, context=None):
        return np.ndarray.__array_wrap__(self.view(type(self)), out_arr, context)

    def __getitem__(self, index):
        index = self._process_index(index)
        return super(Array, self).__getitem__(index)

    def __setitem__(self, index, value):
        index = self._process_index(index)
        return super(Array, self).__setitem__(index, value)

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
