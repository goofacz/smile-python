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

from smile.array import Array


def _get__base_column_names():
        column_names = {
            "position_dimensions": 0,
            "position_x": 1,
            "position_y": 2,
            "position_z": 3,
            "begin_true_position_x": 4,
            "begin_true_position_y": 5,
            "begin_true_position_z": 6,
            "end_true_position_x": 7,
            "end_true_position_y": 8,
            "end_true_position_z": 9,
            "mac_address": 10
        }

        column_names["position_2d"] = (column_names["position_x"],
                                       column_names["position_y"])

        column_names["position_3d"] = (column_names["position_x"],
                                       column_names["position_y"],
                                       column_names["position_z"])

        column_names["begin_true_position_2d"] = (column_names["begin_true_position_x"],
                                                  column_names["begin_true_position_y"])

        column_names["begin_true_position_3d"] = (column_names["begin_true_position_x"],
                                                  column_names["begin_true_position_y"],
                                                  column_names["begin_true_position_z"])

        column_names["end_true_position_2d"] = (column_names["end_true_position_x"],
                                                column_names["end_true_position_y"])

        column_names["end_true_position_3d"] = (column_names["end_true_position_x"],
                                                column_names["end_true_position_y"],
                                                column_names["end_true_position_z"])

        return column_names


class Result:
    def __init__(self):
        self.position_dimensions = None
        self.position_x = None
        self.position_y = None
        self.position_z = None
        self.begin_true_position_x = None
        self.begin_true_position_y = None
        self.begin_true_position_z = None
        self.end_true_position_x = None
        self.end_true_position_y = None
        self.end_true_position_z = None
        self.mac_address = None

    def to_tuple(self):
        result = (self.position_dimensions,
                  self.position_x,
                  self.position_y,
                  self.position_z,
                  self.begin_true_position_x,
                  self.begin_true_position_y,
                  self.begin_true_position_z,
                  self.end_true_position_x,
                  self.end_true_position_y,
                  self.end_true_position_z,
                  self.mac_address)

        if not all(element is not None for element in result):
            raise ValueError('Result tuple cannot contain None values')

        return result


class Results(Array):
    __column_names = _get__base_column_names()

    def __new__(cls, input_array):
        input_array = [row.to_tuple() for row in input_array]
        instance = super(Results, cls).__new__(cls, input_array, Results.__column_names, None)

        unique = np.unique(instance['position_dimensions'])
        if unique.shape != (1,):
            raise ValueError('Array cannot store 2d and 3D positions at the same time')

        return instance

