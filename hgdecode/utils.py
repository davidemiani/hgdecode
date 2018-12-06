import sys
import datetime
from os import getcwd
from os import system
from os import makedirs
from sys import platform
from os.path import join
from os.path import exists
from logging import DEBUG
from logging import getLogger
from logging import basicConfig

log = getLogger(__name__)


def dash_printer(input_string='', manual_input=32):
    print('-' * len(input_string) + '-' * manual_input)


def print_manager(input_string='',
                  print_style='normal',
                  top_return=None,
                  bottom_return=None):
    if top_return is not None:
        print('\n' * (top_return - 1))

    if print_style == 'normal':
        log.info(input_string)
    elif print_style == 'top-dashed':
        dash_printer(input_string)
        log.info(input_string)
        print_manager.last_print = input_string
    elif print_style == 'bottom-dashed':
        log.info(input_string)
        dash_printer(input_string)
        print_manager.last_print = input_string
    elif print_style == 'double-dashed':
        dash_printer(input_string)
        log.info(input_string)
        dash_printer(input_string)
        print_manager.last_print = input_string
    elif print_style == 'last':
        log.info(input_string)
        dash_printer(print_manager.last_print)
        print_manager.last_print = input_string

    if bottom_return is not None:
        print('\n' * (bottom_return - 1))


def datetime_dir_format():
    now = datetime.datetime.now()
    second = str(now.second)
    minute = str(now.minute)
    hour = str(now.hour)
    day = str(now.day)
    month = str(now.month)
    year = str(now.year)
    if len(second) == 1:
        second = '0' + second
    if len(minute) == 1:
        minute = '0' + minute
    if len(hour) == 1:
        hour = '0' + hour
    if len(day) == 1:
        day = '0' + day
    if len(month) == 1:
        month = '0' + month
    return '_'.join(['-'.join([year, month, day]),
                     '-'.join([hour, minute, second])])


def touch_dir(directory):
    if not exists(directory):
        makedirs(directory)


def create_log(results_dir=None, subject_id=1, learning_type='ml'):
    # if not specified, setting '$current_directory/results' as results_dir
    if results_dir is None:
        results_dir = join(getcwd(), 'results')

    # setting subject_id_str
    subject_id_str = str(subject_id)
    if len(subject_id_str) == 1:
        subject_id_str = '0' + subject_id_str

    # setting log_file_dir
    log_file_dir = join(results_dir,
                        learning_type,
                        datetime_dir_format(),
                        subject_id_str)

    # touching log_file_dir
    touch_dir(log_file_dir)

    # setting log_file_name
    log_file_name = 'log.bin'

    # setting log_file_path
    log_file_path = join(log_file_dir, log_file_name)

    # creating the log file
    sys.stdout = open(log_file_path, 'w')

    # opening it using system commands
    if platform == 'linux':
        system('xdg-open ' + log_file_path.replace(' ', '\ '))
    elif platform == 'darwin':  # macOSX
        system('open ' + log_file_path.replace(' ', '\ '))
    else:
        print('platform {:s} still not supported'.format(platform))

    # setting the logging object configuration (created as a global variable)
    basicConfig(
        format='%(asctime)s | %(levelname)s: %(message)s',
        level=DEBUG,
        stream=sys.stdout
    )

    return log
