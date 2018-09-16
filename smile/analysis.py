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


def get_default_groupby_columns():
    return [
        'mac_address',
        'reference_position_x', 'reference_position_y', 'reference_position_z'
    ]


def compute_basic_statistics(results, groupby_columns=None):
    if not groupby_columns:
        groupby_columns = get_default_groupby_columns()

    statistics = {
        'min': np.min,
        'max': np.max,
        'mean': np.mean,
        'std': np.std,
        'var': np.var,
        '25%': lambda values: np.percentile(values, 25),
        '50%': lambda values: np.percentile(values, 50),
        '75%': lambda values: np.percentile(values, 75),
    }

    aggregate_functions = {
        'position_x': statistics,
        'position_y': statistics,
        'position_z': statistics,
    }

    grouped_results = results.groupby(groupby_columns)
    aggregated_results = grouped_results.aggregate(aggregate_functions)
    general_counts = pd.DataFrame(grouped_results.size(), index=aggregated_results.index, dtype=np.int64,
                                  columns=pd.MultiIndex.from_tuples([('positions', 'count')]))
    aggregated_results = aggregated_results.join(general_counts)
    return aggregated_results.reset_index()