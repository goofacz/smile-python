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
import pandas as pd


def get_results_columns():
    return [
        'mac_address',
        'position_dimensions',
        'position_x',
        'position_y',
        'position_z',
        'begin_true_position_x',
        'begin_true_position_y',
        'begin_true_position_z',
        'end_true_position_x',
        'end_true_position_y',
        'end_true_position_z',
    ]


def get_reference_position_columns():
    return [
        'reference_position_x',
        'reference_position_y',
        'reference_position_z',
    ]


def _average_reference_position(begin_positions, end_positions):
    output = (begin_positions.values + end_positions.values) / 2.
    return pd.DataFrame(output, columns=get_reference_position_columns())


def create_results(results, columns=None, reference_positions_function=None):
    if not columns:
        columns = get_results_columns()
    if not reference_positions_function:
        reference_positions_function = _average_reference_position

    results = pd.DataFrame([row.to_tuple() for row in results], columns=columns, dtype=np.float64)
    if len(results.position_dimensions.unique()) != 1:
        raise ValueError('Position dimensions have to be the same for all results')

    begin_positions = results[['begin_true_position_x', 'begin_true_position_y','begin_true_position_z']]
    end_positions = results[['end_true_position_x', 'end_true_position_y','end_true_position_z']]
    reference_positions = reference_positions_function(begin_positions, end_positions)
    return results.join(reference_positions)

