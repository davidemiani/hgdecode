import csv
import sys
import datetime
import logging as log
from os import system
from os import listdir
from os import makedirs
from sys import platform
from os.path import join
from os.path import exists

now_dir = ''


def dash_printer(input_string='', manual_input=32):
    sys.stdout.write('-' * len(input_string) + '-' * manual_input + '\n')


def print_manager(input_string='',
                  print_style='normal',
                  top_return=None,
                  bottom_return=None):
    # top return
    if top_return is not None:
        sys.stdout.write('\n' * top_return)

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
        sys.stdout.write('\n' * bottom_return)


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


def create_log(results_dir,
               learning_type='ml',
               algorithm_or_model_name='FBCSP_rLDA',
               subject_id=1,
               output_on_file=False,
               use_last_result_directory=False):
    # getting now_dir from global
    global now_dir

    # setting temporary results directory
    results_dir = join(results_dir,
                       learning_type,
                       algorithm_or_model_name)

    # setting now_dir if necessary
    if len(now_dir) is 0:
        if use_last_result_directory is True:
            dirs_in_folder = listdir(results_dir)
            dirs_in_folder.sort()
            now_dir = dirs_in_folder[-1]
        else:
            now_dir = datetime_dir_format()

    # setting log_file_dir
    log_file_dir = join(results_dir, now_dir)

    # setting subject_id_str
    if type(subject_id) is str:
        subject_str = subject_id
    else:
        subject_str = str(subject_id)
        if len(subject_str) == 1:
            subject_str = '0' + subject_str
        subject_str = 'subj' + subject_str

    # setting subject_results_dir
    subject_results_dir = join(log_file_dir, subject_str)

    # touching directories
    touch_dir(log_file_dir)
    touch_dir(subject_results_dir)

    if output_on_file is True:
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
            sys.stdout.write('platform {:s} still not supported'.format(
                platform))

        # setting the logging object configuration
        log.basicConfig(
            format='%(asctime)s | %(levelname)s: %(message)s',
            filemode='w',
            stream=sys.stdout,
            level=log.DEBUG
        )
    else:
        # setting the logging object configuration
        log.basicConfig(
            format='%(asctime)s | %(levelname)s: %(message)s',
            level=log.DEBUG,
            stream=sys.stdout
        )

    # printing current cycle information
    print_manager(
        '{} with {} on subject {}'.format(
            learning_type.upper(),
            algorithm_or_model_name,
            subject_id
        ),
        'double-dashed',
        bottom_return=1
    )


def csv_manager(csv_path, line):
    if not exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerows([line])
    else:
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows([line])
