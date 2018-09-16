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

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as npla

_position_columns = ['position_x', 'position_y', 'position_z']
_reference_position_columns = ['reference_position_x', 'reference_position_y', 'reference_position_z']


def plot_absolute_position_error_histogram(results, mac_addresses=None, reference_positions=None):
    if mac_addresses:
        results = results[results.mac_address.isin(mac_addresses)]
    if reference_positions:
        results = results[results[_reference_position_columns].isin(mac_addresses)]

    position_errors = npla.norm(results[_reference_position_columns].values - results[_position_columns].values, axis=1)

    plt.hist(position_errors)
    plt.title('Histogram of absolute error values')
    plt.xlabel('Position error [m]')
    plt.ylabel('Number of localization results')
    plt.grid(True)
    plt.show()


def plot_absolute_position_error_surface(results, anchors=None, mac_addresses=None, reference_positions=None):
    if mac_addresses:
        results = results[results.mac_address.isin(mac_addresses)]
    if reference_positions:
        results = results[results[_reference_position_columns].isin(mac_addresses)]

    reference_positions = results[_reference_position_columns]
    position_errors = npla.norm(reference_positions.values - results[_position_columns].values, axis=1)

    x, y = np.meshgrid(reference_positions.reference_position_x.unique(),
                       reference_positions.reference_position_x.unique(),
                       indexing='xy')
    z = np.zeros(x.shape)

    # Assign position errors (z) to correct node coordinates (x, y)
    for i in range(0, len(reference_positions)):
        tmp_true_position = reference_positions.iloc[i, :]
        tmp_position_error = position_errors[i]

        tmp_z_indices = np.where(np.logical_and(x == tmp_true_position[0], y == tmp_true_position[1]))
        z[tmp_z_indices] = tmp_position_error

    # Plot errors
    plt.pcolormesh(x, y, z)

    x = x.flatten().tolist()
    y = y.flatten().tolist()

    # Plot anchors' positions
    if anchors is not None:
        for i in range(anchors.shape[0]):
            anchor_x = anchors[i, "position_x"]
            anchor_y = anchors[i, "position_y"]
            plt.plot(anchor_x, anchor_y, "ro", markersize=10)

            x.append(anchor_x)
            y.append(anchor_y)

    # Set canvas limits
    x_min = min(x) - 2.5
    x_max = max(x) + 2.5
    y_min = min(y) - 2.5
    y_max = max(y) + 2.5
    plt.xlim(x_min, x_max)
    plt.ylim(y_max, y_min)

    axis = plt.gca()
    axis.xaxis.set_ticks_position('top')
    axis.xaxis.set_label_position('top')
    axis.minorticks_on()
    plt.colorbar().set_label('Error value [m]')
    plt.axis('equal')
    plt.grid()
    plt.title('Map of absolute position errors', y=-0.1)
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.show()


def plot_absolute_position_error_cdf(results, mac_addresses=None, reference_positions=None):
    if mac_addresses:
        results = results[results.mac_address.isin(mac_addresses)]
    if reference_positions:
        results = results[results[_reference_position_columns].isin(mac_addresses)]

    position_errors = npla.norm(results[_reference_position_columns].values - results[_position_columns].values, axis=1)
    position_errors = np.sort(position_errors)
    n = np.array(range(position_errors.size)) / np.float(position_errors.size)

    plt.plot(position_errors, n)
    plt.grid()
    plt.title('CDF of absolute position error')
    plt.xlabel('Error [m]')
    plt.show()