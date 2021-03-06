import os
from csv import reader
from hgdecode.utils import get_path
from hgdecode.utils import check_significant_digits

"""
SET HERE YOUR PARAMETERS
"""
# to find file parameters
results_dir = None
learning_type = 'dl'
algorithm_or_model_name = None
epoching = '-1500_500'
fold_type = 'single_subject'
n_folds = 12
deprecated = True
balanced_fold = True

# metrics parameter
label = 'Feet'  # Feet, LeftHand, Rest or RightHand
metric_type = 'overall'  # label or overall
metric = 'acc'

"""
GETTING PATHS
"""
# getting folder path
folder_path = get_path(
    results_dir=results_dir,
    learning_type=learning_type,
    algorithm_or_model_name=algorithm_or_model_name,
    epoching=epoching,
    fold_type=fold_type,
    n_folds=n_folds,
    deprecated=deprecated,
    balanced_folds=balanced_fold
)

# getting file_path
file_path = os.path.join(folder_path, 'statistics', 'tables')
if metric_type == 'overall':
    file_path = os.path.join(file_path, metric + '.csv')
else:
    file_path = os.path.join(file_path, label, metric + '.csv')

"""
COMPUTATION START HERE
"""
with open(file_path) as f:
    csv = list(reader(f))

n_folds = len(csv[0]) - 2
columns = ['&\\textbf{' + str(x + 1) + '}\n' for x in range(n_folds)]

output = '\\begin{table}[H]\n\\footnotesize\n\\centering\n\\begin{tabular}' + \
         '{|c|' + 'c' * n_folds + '|cc|}\n\\hline\n' + \
         '&\multicolumn{' + str(n_folds) + '}{c|}{\\textbf{fold}}& ' + \
         '&\n\\\\\n\\textbf{subj}\n'
for head in columns:
    output += head
output += '&\\textbf{mean}\n&\\textbf{std}\n\\\\\n\hline\hline\n'

# removing header
csv = csv[1:]
total_m = 0
total_s = 0

for idx, current_row in enumerate(csv):
    if idx % 2 is 0:
        output += '\\rowcolor[gray]{.9}\n'
    else:
        output += '\\rowcolor[gray]{.8}\n'
    output += '\\textbf{' + str(idx + 1) + '}\n'
    for idy, current_col in enumerate(current_row):
        output += '&' + check_significant_digits(current_col) + '\n'
    output += '\\\\\n'
    total_m += float(current_row[-2])
    total_s += float(current_row[-1])
total_m /= len(csv)
total_s /= len(csv)
total_m = check_significant_digits(total_m)
total_s = check_significant_digits(total_s)

caption = learning_type + ' ' + metric + ' ' + \
          epoching.replace('_', ',') + ' ' + str(n_folds) + ' fold'
output += '\\hline\n\\multicolumn{' + str(n_folds + 1) + '}' + \
          '{|r|}{\\textbf{media totale}}\n&' + total_m + '\n&' + \
          total_s + '\n\\\\\n\\hline\n\\end{tabular}\n' + \
          '\\caption{' + caption + '}\n\\label{' + caption + '}\n' + \
          '\\end{table}'
print(output)
