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

from os.path import expanduser, abspath
import numpy as np

from smile.array import Array


class Frames(Array):
    def __init__(self, *args):
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

        super(Frames, self).__init__(column_names)

    @staticmethod
    def load_csv(file_path):
        """
        Loads frames from CSV file.

        :param file_path: Path to CSV file
        """
        converters = Frames._get_default_converters()
        if isinstance(file_path, str):
            file_path = abspath(expanduser(file_path))
        return Frames(np.loadtxt(file_path, delimiter=',', converters=converters, ndmin=2))

    @staticmethod
    def _get_default_converters():
        return {1: lambda value: hash(value)}
