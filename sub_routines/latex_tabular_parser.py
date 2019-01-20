from csv import reader
from numpy import round
from os.path import join

main_folder = '2019-01-15_01-54-25'
loss_acc = 'loss'
file_path = join(main_folder, 'statistics', 'tables', loss_acc + '.csv')

with open(file_path) as f:
    csv = list(reader(f))

output = '\\begin{table}[H]\n\\centering\n\\begin{tabular}' + \
         '{|c|cccccccc|c|}\n\\hline\n' + \
         '&\multicolumn{8}{c|}{\\textbf{fold}}&\n\\\\\n' + \
         '\\textbf{subj}\n&\\textbf{1}\n&\\textbf{2}\n&\\textbf{3}\n' + \
         '&\\textbf{4}\n&\\textbf{5}\n&\\textbf{6}\n&\\textbf{7}\n' + \
         '&\\textbf{8}\n&\\textbf{mean}\n\\\\\n\hline\hline\n'

# removing header
csv = csv[1:]
total = 0

for idx, current_row in enumerate(csv):
    if idx % 2 is 0:
        output += '\\rowcolor[gray]{.9}\n'
    else:
        output += '\\rowcolor[gray]{.8}\n'
    output += '\\textbf{' + str(idx + 1) + '}\n'
    for idy, current_col in enumerate(current_row):
        temp = str(round(float(current_col), 3))
        temp += '0' * (5 - len(temp))
        if temp[0] is '0':
            temp = temp[1:]
        else:
            temp = temp[0:-1]
        output += '&' + temp + '\n'
    output += '\\\\\n'
    total += float(current_row[-1])
total /= len(csv)
total = str(round(total, 3))
total += '0' * (5 - len(total))
if total[0] is '0':
    total = total[1:]
else:
    total = total[0:-1]

output += '\\hline\n\\multicolumn{9}{|r|}{\\textbf{media totale}}\n&' + \
          total + '\n\\\\\n\\hline\n\\end{tabular}\n' + \
          '\\caption{' + loss_acc + '}\n\\label{' + loss_acc + '}\n' + \
          '\\end{table}'
print(output)
