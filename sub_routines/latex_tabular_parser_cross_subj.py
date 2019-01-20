import os
from csv import reader
from hgdecode.utils import check_significant_digits

results_dir = '/Users/davidemiani/OneDrive - Alma Mater Studiorum ' \
              'Università di Bologna/TesiMagistrale_DavideMiani/' \
              'results/hgdecode'
learning_type = 'ml'  # dl or ml
algo_or_model_name = 'FBCSP_rLDA'  # DeepConvNet or FBCSP_rLDA
datetime = '2019-01-20_11-57-47'
epoch_ival_ms = '-500, 4000'  # str type
tables_dir = os.path.join(results_dir,
                          learning_type,
                          algo_or_model_name,
                          datetime,
                          'statistics',
                          'tables')
label_names = ['RightHand', 'LeftHand', 'Rest', 'Feet']

# pre-allocating csv
csv = []

# getting accuracy
with open(os.path.join(tables_dir, 'acc.csv')) as f:
    temp = list(reader(f))
    temp = temp[1]
    csv.append(temp)

# getting precision
for label in label_names:
    with open(os.path.join(tables_dir, label, 'prec.csv')) as f:
        temp = list(reader(f))
        temp = temp[1]
        csv.append(temp)

# getting f1 score
for label in label_names:
    with open(os.path.join(tables_dir, label, 'f1.csv')) as f:
        temp = list(reader(f))
        temp = temp[1]
        csv.append(temp)

# transposing csv
csv = list(map(list, zip(*csv)))

# cutting away last two rows (mean and std)
csv_2 = csv[-2:]
csv = csv[0:-2]

output = '\\begin{table}[H]\n\\footnotesize\n\\centering\n\\begin{tabular}' + \
         '{|c|ccccccccc|}\n\\hline\n' + \
         '\\textbf{fold}\n&' + \
         '\\textbf{acc}\n&' + \
         '\\textbf{pr 1}\n&' + \
         '\\textbf{pr 2}\n&' + \
         '\\textbf{pr 3}\n&' + \
         '\\textbf{pr 4}\n&' + \
         '\\textbf{f1 1}\n&' + \
         '\\textbf{f1 2}\n&' + \
         '\\textbf{f1 3}\n&' + \
         '\\textbf{f1 4}\n' + \
         '\\\\\n\\hline\\hline\n'

for idx, current_row in enumerate(csv):
    if idx % 2 is 0:
        output += '\\rowcolor[gray]{.9}\n'
    else:
        output += '\\rowcolor[gray]{.8}\n'
    output += '\\textbf{' + str(idx + 1) + '}\n'
    for idy, current_col in enumerate(current_row):
        output += '&' + check_significant_digits(current_col) + '\n'
    output += '\\\\\n'
output += '\\hline\n'

for idx, current_row in enumerate(csv_2):
    if idx == 0:
        output += '\\textbf{mean}\n'
    else:
        output += '\\textbf{std}\n'
    for idy, current_col in enumerate(current_row):
        output += '&' + check_significant_digits(current_col) + '\n'
    output += '\\\\\n'
output += '\\hline\n'

caption = learning_type + ' cross-subject validation ' + epoch_ival_ms
output += '\\end{tabular}\n' + \
          '\\caption{' + caption + '}\n\\label{' + caption + '}\n' + \
          '\\end{table}'
print(output)
