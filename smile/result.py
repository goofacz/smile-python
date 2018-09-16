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


class Result:
    def __init__(self):
        self.mac_address = None
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
        self.reference_position_x = None
        self.reference_position_y = None
        self.reference_position_z = None

    def to_tuple(self):
        if self.position_dimensions not in (1, 2, 3):
            raise ValueError('Result\'s position have to me in 1, 2 or 3 dimensions')

        result = (
            np.int(self.mac_address),
            np.int(self.position_dimensions),
            np.float64(self.position_x),
            np.float64(self.position_y),
            np.float64(self.position_z),
            np.float64(self.begin_true_position_x),
            np.float64(self.begin_true_position_y),
            np.float64(self.begin_true_position_z),
            np.float64(self.end_true_position_x),
            np.float64(self.end_true_position_y),
            np.float64(self.end_true_position_z),
            np.float64(self.reference_position_x),
            np.float64(self.reference_position_y),
            np.float64(self.reference_position_z),
        )

        if not not any(np.isnan(result)):
            raise ValueError('Result tuple cannot contain None values')

        return result
