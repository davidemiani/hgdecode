import os
from csv import reader
from hgdecode.utils import check_significant_digits

results_dir = '/Users/davidemiani/OneDrive - Alma Mater Studiorum ' \
              'Università di Bologna/TesiMagistrale_DavideMiani/' \
              'results/hgdecode'
learning_type = 'dl'  # dl or ml
if learning_type == 'ml':
    algo_or_model_name = 'FBCSP_rLDA'
else:
    algo_or_model_name = 'DeepConvNet'
datetime = '2019-01-27_15-07-56'
label = 'Feet'  # Feet, LeftHand, Rest or RightHand
metric_type = 'overall'  # label or overall
metric = 'acc'
epoch_ival_ms = '-500,4000'  # str type
file_path = os.path.join(results_dir,
                         learning_type,
                         algo_or_model_name,
                         datetime,
                         'statistics',
                         'tables')
if metric_type == 'overall':
    file_path = os.path.join(file_path, metric + '.csv')
else:
    file_path = os.path.join(file_path, label, metric + '.csv')

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

caption = learning_type + ' ' + metric + ' ' + epoch_ival_ms
output += '\\hline\n\\multicolumn{' + str(n_folds + 1) + '}' + \
          '{|r|}{\\textbf{media totale}}\n&' + total_m + '\n&' + \
          total_s + '\n\\\\\n\\hline\n\\end{tabular}\n' + \
          '\\caption{' + caption + '}\n\\label{' + caption + '}\n' + \
          '\\end{table}'
print(output)
