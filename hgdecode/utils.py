import sys
import datetime
import logging as log
from os import getcwd
from os import system
from os import makedirs
from sys import platform
from os.path import join
from os.path import exists

now_dir = ''


def dash_printer(input_string='', manual_input=32):
    print('-' * len(input_string) + '-' * manual_input)


def print_manager(input_string='',
                  print_style='normal',
                  top_return=None,
                  bottom_return=None):
    # top return
    if top_return is not None:
        print('\n' * (top_return - 1))

    # input_string
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

    # bottom return
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
    if exists(directory):
        return True
    else:
        makedirs(directory)
        return False


def touch_file(file_path):
    if exists(file_path):
        return True
    else:
        return False


def create_log(results_dir=None,
               learning_type='ml',
               algorithm_or_model_name='FBCSP_rLDA',
               subject_id=1):
    # getting now_dir from global
    global now_dir
    if len(now_dir) == 0:
        now_dir = datetime_dir_format()

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
                        algorithm_or_model_name,
                        now_dir)

    # setting subject_results_dir
    subject_results_dir = join(log_file_dir, subject_id_str)

    # touching directories
    touch_dir(log_file_dir)
    touch_dir(subject_results_dir)

    # setting log_file_name
    log_file_name = 'log.bin'

    # setting log_file_path
    log_file_path = join(log_file_dir, log_file_name)

    # creating and using log file as output only if it does not exist already
    if touch_file(log_file_path):
        pass
    else:
        # creating the log file
        sys.stdout = open(log_file_path, 'w')

        # opening it using system commands
        if platform == 'linux':
            system('xdg-open ' + log_file_path.replace(' ', '\ '))
        elif platform == 'darwin':  # macOSX
            system('open ' + log_file_path.replace(' ', '\ '))
        else:
            print('platform {:s} still not supported'.format(platform))

        # setting the logging object configuration
        log.basicConfig(
            format='%(asctime)s | %(levelname)s: %(message)s',
            filemode='w',
            stream=sys.stdout,
            level=log.DEBUG
        )
