import os
from csv import reader
from numpy import array
from scipy.stats import ttest_rel
from hgdecode.utils import get_path

"""
TRAINING 1
"""
results_dir = None
learning_type = 'dl'
algorithm_or_model_name = None
epoching = '-500_4000'
fold_type_1 = 'single_subject'
n_folds_list = [12]  # must be a list of integer
deprecated = False
balanced_folds = True
folder_paths_1 = [
    get_path(
        results_dir=results_dir,
        learning_type=learning_type,
        algorithm_or_model_name=algorithm_or_model_name,
        epoching=epoching,
        fold_type=fold_type_1,
        n_folds=x,
        deprecated=deprecated,
        balanced_folds=balanced_folds
    )
    for x in n_folds_list
]

"""
TRAINING 2
"""
results_dir = None
learning_type = 'dl'
algorithm_or_model_name = None
epoching = '-1000_1000'
fold_type_2 = 'single_subject'
n_folds_list = [12]  # must be a list of integer
deprecated = False
balanced_folds = True
folder_paths_2 = [
    get_path(
        results_dir=results_dir,
        learning_type=learning_type,
        algorithm_or_model_name=algorithm_or_model_name,
        epoching=epoching,
        fold_type=fold_type_2,
        n_folds=x,
        deprecated=deprecated,
        balanced_folds=balanced_folds
    )
    for x in n_folds_list
]

"""
T-TESTING
"""
for training_1, training_2 in zip(folder_paths_1, folder_paths_2):
    # loading training_1 accuracies
    training_1_acc_csv_path = os.path.join(training_1,
                                           'statistics', 'tables', 'acc.csv')
    with open(training_1_acc_csv_path) as f:
        training_1_csv = list(reader(f))
    if fold_type_1 == 'cross_subject':
        training_1_accs = array(list(map(float, training_1_csv[1][:-2])))
    else:
        training_1_accs = array([
            float(training_1_csv[x][-2]) for x in range(1, len(training_1_csv))
        ])

    # loading training_2 accuracies
    training_2_acc_csv_path = os.path.join(training_2,
                                           'statistics', 'tables', 'acc.csv')
    with open(training_2_acc_csv_path) as f:
        training_2_csv = list(reader(f))
    if fold_type_2 == 'cross_subject':
        training_2_accs = array(list(map(float, training_2_csv[1][:-2])))
    else:
        training_2_accs = array([
            float(training_2_csv[x][-2]) for x in range(1, len(training_2_csv))
        ])

    # running t-test
    statistic, p_value = ttest_rel(training_1_accs, training_2_accs)
    print(p_value * 100)
