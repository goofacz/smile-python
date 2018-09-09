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
        "mac_address": 0,
        "position_x": 1,
        "position_y": 2,
        "position_z": 3
    }

    column_names["position_2d"] = (column_names["position_x"],
                                   column_names["position_y"])

    column_names["position_3d"] = (column_names["position_x"],
                                   column_names["position_y"],
                                   column_names["position_z"])

    return column_names


class Nodes(Array):
    _column_names = _get__base_column_names()

    def __new__(cls, input_array):
        return super(Nodes, cls).__new__(cls, input_array, Nodes._column_names, None)