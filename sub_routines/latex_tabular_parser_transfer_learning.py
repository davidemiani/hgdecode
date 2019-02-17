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
p_flag = False  # if true, it will print p value too.

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
           ['mean', 'std', '$\\Delta_{\\textbf{\\%}}$', '$p$']]
header = ['ML', 'DL', '4', '8', '16', '32', '64', '128']
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
if p_flag is False:
    rows.pop()

"""
CREATING LATEX TABULAR CODE
"""
# pre-allocating output
output = ''

# opening table
output += '\\begin{table}[H]\n\\footnotesize\n\\centering\n'
output += '\\begin{tabular}{|c|M{1.4cm}M{1.4cm}|'
output += 'c' * len(train_trials_list) + '|}\n'
output += '\\hline\n&\multicolumn{2}{c|}{\\textbf{cross-soggetto}}\n'
output += '&\multicolumn{' + str(len(train_trials_list)) + '}{c|}'
output += '{\\textbf{transfer learning}}\n\\\\\n'

# first row is an header
for idx, col in enumerate(rows[0]):
    if idx == 0:
        output += '\\textbf{' + col + '}\n'
    else:
        output += '&\\textbf{' + col + '}\n'
output += '\\\\\n\\hline\n\\hline\n'

# creating iterator and jumping the first element (header)
iterator = iter(rows)
next(iterator)

for idx, row in enumerate(iterator):
    if idx % 2 == 0:
        output += '\\rowcolor[gray]{.9}\n'
    else:
        output += '\\rowcolor[gray]{.8}\n'
    for idy, col in enumerate(row):
        if idy == 0:
            output += '\\textbf{' + col + '}\n'
        else:
            output += '&' + col + '\n'
    output += '\\\\\n'
    if idx == n_subjs - 1:
        output += '\\hline\n\\hline\n'

output += '\\hline\n\\end{tabular}'
output += '\n\\caption{tl table}\n\\label{tl table}\n'
output += '\\end{table}'
print(output)
