import os
import numpy as np
from csv import reader


def check_significant_digits(num):
    num = float(num)
    if num < 0.01:  # from 0.009999
        num = np.round(num, 5)
    elif num < 0.1:  # from 0.09999
        num = np.round(num, 4)
    else:
        num = np.round(num, 3)
    num = num * 100
    if num < 1:
        num = np.round(num, 3)
        num = str(num)
        num += '0' * (5 - len(num))
    elif num < 10:
        num = np.round(num, 2)
        num = str(num)
        num += '0' * (4 - len(num))
    elif num == 100:
        num = '100'
    else:
        num = np.round(num, 1)
        num = str(num)
    return num


results_dir = '/Users/davidemiani/OneDrive - Alma Mater Studiorum ' \
              'Università di Bologna/TesiMagistrale_DavideMiani/' \
              'results/hgdecode'
learning_type = 'ml'  # dl or ml
algo_or_model_name = 'FBCSP_rLDA'  # DeepConvNet or FBCSP_rLDA
datetime = '2019-01-20_05-23-10'
label = 'Feet'  # Feet, LeftHand, Rest or RightHand
metric_type = 'overall'  # label or overall
metric = 'acc'
epoch_ival_ms = '-500, 4000'  # str type
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

output = '\\begin{table}[H]\n\\footnotesize\n\\centering\n\\begin{tabular}' + \
         '{|c|cccccccc|cc|}\n\\hline\n' + \
         '&\multicolumn{8}{c|}{\\textbf{fold}}& &\n\\\\\n' + \
         '\\textbf{subj}\n&\\textbf{1}\n&\\textbf{2}\n&\\textbf{3}\n' + \
         '&\\textbf{4}\n&\\textbf{5}\n&\\textbf{6}\n&\\textbf{7}\n' + \
         '&\\textbf{8}\n&\\textbf{mean}\n&\\textbf{std}\n\\\\\n\hline\hline\n'

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
output += '\\hline\n\\multicolumn{9}{|r|}{\\textbf{media totale}}\n&' + \
          total_m + '\n&' + total_s + '\n\\\\\n\\hline\n\\end{tabular}\n' + \
          '\\caption{' + caption + '}\n\\label{' + caption + '}\n' + \
          '\\end{table}'
print(output)
