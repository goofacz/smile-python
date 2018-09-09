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

from smile.array import Array


def _get__base_column_names():
    column_names = {
        "node_mac_address": 0,
        "direction": 1,
        "begin_clock_timestamp": 2,
        "begin_simulation_timestamp": 3,
        "begin_true_position_x": 4,
        "begin_true_position_y": 5,
        "begin_true_position_z": 6,
        "end_clock_timestamp": 7,
        "end_simulation_timestamp": 8,
        "end_true_position_x": 9,
        "end_true_position_y": 10,
        "end_true_position_z": 11,
        "source_mac_address": 12,
        "destination_mac_address": 13,
        "sequence_number": 14
    }

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


def _get_base_column_converters():
    return {1: lambda value: hash(value)}


class Frames(Array):
    __column_names = _get__base_column_names()
    __column_converters = _get_base_column_converters()

    def __new__(cls, input_array):
        return super(Frames, cls).__new__(cls, input_array, Frames.__column_names,
                                          column_converters=Frames.__column_converters)