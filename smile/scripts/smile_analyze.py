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

import argparse
import importlib
import sys

import smile.analysis as sa
import smile.visualization as sv
from smile.platform.json_configuration import *

__EXPERIMENTS_DIRECTORY = 'experiments'
__ANALYSER_DIRECTORY = 'smile_algorithm'

def __import_method_module(method_path):
    sys.path.append(method_path)
    return importlib.import_module(f'{__ANALYSER_DIRECTORY}.algorithm')


def __parse_cmd_arguments():
    parser = argparse.ArgumentParser(description='Run SMILe analysis')
    parser.add_argument('method_path', type=str, nargs=1, help='Path to method analysis.')
    parser.add_argument('experiment_name', type=str, nargs=1, help='Experiment name.')
    parser.add_argument('results_path', type=str, nargs=1, help='Path to simulation results.')
    arguments = parser.parse_args()

    method_path = arguments.method_path[0]
    experiment_name = arguments.experiment_name[0]
    results_path = arguments.results_path[0]

    # Validate method_path
    if not os.path.isdir(method_path):
        raise RuntimeError(f'{method_path} is not a directory or it doesn\t exist.')

    analyser_path = os.path.join(method_path, __ANALYSER_DIRECTORY)
    if not os.path.isdir(analyser_path):
        raise RuntimeError(f'{method_path} has to contain {__ANALYSER_DIRECTORY} directory.')

    experiments_path = os.path.join(method_path, __EXPERIMENTS_DIRECTORY)
    if not os.path.isdir(experiments_path):
        raise RuntimeError(f'{method_path} has to contain {__EXPERIMENTS_DIRECTORY} directory.')

    # Validate results_path
    if not os.path.isdir(results_path):
        raise RuntimeError(f'{results_path} is not a directory or it doesn\t exist.')

    return method_path, experiment_name, results_path


def __get_json_file_path(method_path, experiment_name):
    path = os.path.join(method_path, __EXPERIMENTS_DIRECTORY, experiment_name, 'algorithm.json')
    if not os.path.isfile(path):
        raise RuntimeError(f'{path} is not valid path to JSON file')

    return path


def main():
    method_path, experiment_name, results_path = __parse_cmd_arguments()
    json_file_path = __get_json_file_path(method_path, experiment_name)

    configuration = JsonConfiguration(json_file_path)
    method_module = __import_method_module(method_path)

    analyser = method_module.Algorithm(configuration)
    results, anchors = analyser.run_offline(results_path)

    unique_results = sa.squeeze_results(results)
    sv.plot_absolute_position_error_cdf(unique_results)
    sv.plot_absolute_position_error_surface(unique_results, anchors)
    sv.plot_absolute_position_error_histogram(unique_results)


if __name__ == '__main__':
    main()