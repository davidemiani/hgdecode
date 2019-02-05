import csv
import sys
import datetime
import numpy as np
import logging as log
from os import getcwd
from os import system
from os import listdir
from os import makedirs
from sys import platform
from pickle import dump
from os.path import join
from os.path import exists
from os.path import dirname
from sklearn.metrics import confusion_matrix

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


def listdir2(path):
    l = listdir(path)
    for c_dir in l:
        if c_dir == '.DS_Store':
            l.remove('.DS_Store')
    return l


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

    # returning subject_results_dir, in some case it can be helpful
    return subject_results_dir


def ml_results_saver(exp, subj_results_dir):
    for fold_idx in range(exp.n_folds):
        # computing paths and directories
        fold_str = str(fold_idx + 1)
        if len(fold_str) is not 2:
            fold_str = '0' + fold_str
        fold_str = 'fold' + fold_str
        fold_dir = join(subj_results_dir, fold_str)
        touch_dir(fold_dir)
        file_path = join(fold_dir, 'fold_stats.pickle')

        # getting accuracies
        train_acc = exp.multi_class.train_accuracy[fold_idx]
        test_acc = exp.multi_class.test_accuracy[fold_idx]

        # getting y_true and y_pred for this fold
        train_true = exp.multi_class.train_labels[fold_idx]
        train_pred = exp.multi_class.train_predicted_labels[fold_idx]
        test_true = exp.multi_class.test_labels[fold_idx]
        test_pred = exp.multi_class.test_predicted_labels[fold_idx]

        # computing confusion matrices
        train_conf_mtx = confusion_matrix(train_true, train_pred)
        test_conf_mtx = confusion_matrix(test_true, test_pred)

        # creating results dictionary
        results = {
            'train': {
                'acc': train_acc,
                'conf_mtx': train_conf_mtx.tolist()
            },
            'test': {
                'acc': test_acc,
                'conf_mtx': test_conf_mtx.tolist()
            }
        }

        # saving results
        with open(file_path, 'wb') as f:
            dump(results, f)


def csv_manager(csv_path, line):
    if not exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerows([line])
    else:
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows([line])


def get_metrics_from_conf_mtx(conf_mtx, label_names=None):
    # creating standard label_names if not specified
    if label_names is None:
        label_names = ['label ' + str(x) for x in range(len(conf_mtx))]

    # computing true/false positive/negative from confusion matrix
    TP = np.diag(conf_mtx)
    FP = conf_mtx.sum(axis=0) - TP
    FN = conf_mtx.sum(axis=1) - TP
    TN = conf_mtx.sum() - (FP + FN + TP)

    # parsing to float
    TP = TP.astype(float)
    TN = TN.astype(float)
    FP = FP.astype(float)
    FN = FN.astype(float)

    # computing true positive rate (sensitivity, hit rate, recall)
    TPR = TP / (TP + FN)

    # computing true negative rate (specificity)
    TNR = TN / (TN + FP)

    # computing positive predictive value (precision)
    PPV = TP / (TP + FP)

    # computing negative predictive value
    NPV = TN / (TN + FN)

    # computing false positive rate (fall out)
    FPR = FP / (FP + TN)

    # computing false negative rate
    FNR = FN / (TP + FN)

    # computing false discovery rate
    FDR = FP / (TP + FP)

    # computing f1-score
    F1 = 2 * TP / (2 * TP + FP + FN)

    # computing accuracy on single label
    ACC = (TP + TN) / (TP + FP + FN + TN)

    # computing overall accuracy
    acc = TP.sum() / conf_mtx.sum()

    # pre-allocating metrics_report
    metrics_report = {x: None for x in label_names}
    metrics_report['acc'] = acc

    # filling metrics_report
    for idx, label in enumerate(label_names):
        metrics_report[label] = {
            'TP': TP[idx],
            'TN': TN[idx],
            'FP': FP[idx],
            'FN': FN[idx],
            'TPR': TPR[idx],
            'TNR': TNR[idx],
            'PPV': PPV[idx],
            'NPV': NPV[idx],
            'FPR': FPR[idx],
            'FNR': FNR[idx],
            'FDR': FDR[idx],
            'F1': F1[idx],
            'ACC': ACC[idx]
        }

    # returning metrics_report
    return metrics_report


def check_significant_digits(num):
    num = float(num)
    if num < 0.01:  # from 0.009999
        num = np.round(num, 5)
    elif num < 0.1:  # from 0.09999
        num = np.round(num, 4)
    else:
        num = np.round(num, 3)
    num = num * 100
    if num == 0:
        num = '0'
    elif num < 1:
        num = np.round(num, 3)
        num = str(num)
        num += '0' * (5 - len(num))
        num = num[0:4]
    elif num < 10:
        num = np.round(num, 2)
        num = str(num)
        num += '0' * (4 - len(num))
        num = num[0:4]
    elif num == 100:
        num = '100'
    else:
        num = np.round(num, 1)
        num = str(num)
    return num


def get_subj_str(subj_id):
    return my_formatter(subj_id, 'subj')


def get_fold_str(fold_id):
    return my_formatter(fold_id, 'fold')


def my_formatter(num, name):
    num_str = str(num)
    if len(num_str) == 1:
        num_str = '0' + num_str
    return name + num_str


def get_path(results_dir=None,
             learning_type='dl',
             algorithm_or_model_name=None,
             epoching=(-500, 4000),
             fold_type='single_subject',
             n_folds=2,
             deprecated=False):
    # checking results_dir
    if results_dir is None:
        results_dir = join(dirname(dirname(dirname(getcwd()))), 'results')

    # checking algorithm_or_model_name
    if algorithm_or_model_name is None:
        if learning_type == 'ml':
            algorithm_or_model_name = 'FBCSP_rLDA'
        elif learning_type == 'dl':
            algorithm_or_model_name = 'DeepConvNet'
        else:
            raise ValueError(
                'Invalid learning_type inputed: {}'.format(learning_type)
            )

    # checking epoching
    if epoching.__class__ is tuple or epoching.__class__ is list:
        epoching_str = str(epoching[0]) + '_' + str(epoching[1])
    elif epoching.__class__ is str:
        epoching_str = epoching
    else:
        raise ValueError(
            'Invalid epoching type: {}'.format(epoching.__class__)
        )

    # checking fold_type
    folder = ''
    if fold_type == 'schirrmeister':
        folder += '1'
    elif fold_type == 'single_subject':
        if epoching_str == '-500_4000':
            folder += '2'
        elif epoching_str == '-1000_1000':
            folder += '3'
        elif epoching_str == '-1500_500':
            folder += '4'
        elif epoching_str == '-2000_0':
            folder += '5'
    elif fold_type == 'cross_subject':
        if epoching_str == '-500_4000':
            folder += '6'
        elif epoching_str == '-1000_1000':
            folder += '7'
    elif fold_type == 'transfer_learning':
        folder += '8'
    else:
        raise ValueError(
            'Invalid fold_type: {}'.format(fold_type)
        )
    folder += '_' + fold_type + '_' + epoching_str

    if deprecated is True:
        folder = join('0_deprecated', folder)

    folder_path = join(results_dir,
                       'hgdecode',
                       learning_type,
                       algorithm_or_model_name,
                       folder)

    if (fold_type == 'single_subject') and (epoching_str == '-500_4000'):
        folder_path = join(folder_path, my_formatter(n_folds, 'fold'))

    return join(folder_path, listdir2(folder_path)[0])
