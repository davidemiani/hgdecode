import os
import numpy as np
from csv import reader
from scipy.stats import ttest_rel
from hgdecode.utils import get_path
from hgdecode.utils import check_significant_digits

"""
SET HERE YOUR PARAMETERS
"""
train_trials_list = [4, 8, 16, 32, 64, 128]
reference = 1  # 0 for ML cross, 1 for DL cross, 2 for TL4 ecc.

"""
GETTING PATHS
"""
folder_paths = [
    get_path(
        results_dir=None,
        learning_type=x,
        algorithm_or_model_name=y,
        epoching=(-500, 4000),
        fold_type='cross_subject',
        n_folds=None,
        deprecated=True
    )
    for x, y in zip(['ml', 'dl'], ['FBCSP_rLDA', 'DeepConvNet'])
]

folder_paths += [
    get_path(
        results_dir=None,
        learning_type='dl',
        algorithm_or_model_name='DeepConvNet',
        epoching=(-500, 4000),
        fold_type='transfer_learning',
        n_folds=x,
        deprecated=True
    )
    for x in train_trials_list
]

# getting file_path
csv_paths = [
    os.path.join(x, 'statistics', 'tables', 'acc.csv')
    for x in folder_paths
]

"""
GETTING DATA FROM CSV FILES
"""
subj_data = []
mean_data = []
stdd_data = []
for idx, csv_path in enumerate(csv_paths):
    with open(csv_path) as f:
        csv = list(reader(f))
    csv = csv[1:]
    csv = [
        list(map(float, csv[x]))
        for x in range(len(csv))
    ]
    if idx < 2:
        subj_data.append(csv[0][:-2])
        mean_data.append(csv[0][-2])
        stdd_data.append(csv[0][-1])
    else:
        temp_data = []
        for csv_line in csv:
            temp_data.append(csv_line[-2])
        subj_data.append(temp_data)
        mean_data.append(np.mean(temp_data))
        stdd_data.append(np.std(temp_data))

"""
COMPUTING PERC AND PVAL
"""
perc_data = []
pval_data = []
for idx in range(len(mean_data)):
    if idx is reference:
        perc_data.append(0.)
        pval_data.append(float('nan'))
    else:
        perc_data.append((mean_data[idx] - mean_data[reference]) /
                         mean_data[reference])
        pval_data.append(ttest_rel(subj_data[idx], subj_data[reference])[1])

"""
GENERAL FORMATTING
"""
n_subjs = len(subj_data[0])
columns = [['subj'] + list(map(str, range(1, n_subjs + 1))) +
           ['mean', 'std', '$\\Delta$\\%', '$p_{\\%}$']]
header = [
    'CS ML', 'CS DL',
    'TL 4', 'TL 8', 'TL 16', 'TL 32', 'TL 64', 'TL 128'
]
for idx, head in enumerate(header):
    temp = [check_significant_digits(str(subj_data[idx][x]))
            for x in range(n_subjs)]
    columns.append(
        [head] +
        temp +
        [str(check_significant_digits(mean_data[idx]))] +
        [str(check_significant_digits(stdd_data[idx]))] +
        [str(check_significant_digits(perc_data[idx]))] +
        [str(check_significant_digits(pval_data[idx]))]
    )
rows = list(map(list, zip(*columns)))
