import os
from os.path import join, dirname
import numpy as np
import matplotlib.pyplot as plt
from pylab import savefig
from csv import reader
from hgdecode.utils import get_path, touch_dir

"""
SET HERE YOUR PARAMETERS
"""
ival = (-500, 4000)
frozen_layers_list = [1, 2, 3, 4, 5, -1, -2, -3, -4, -5, 6]

fontsize_1 = 35
fontsize_2 = 27.5
fig_size = (22, 7.5)

"""
GETTING PATHS
"""
folder_paths = [get_path(
    results_dir=None,
    learning_type='dl',
    algorithm_or_model_name=None,
    epoching=ival,
    fold_type='transfer_learning',
    n_folds=128,
    deprecated=False
)]

folder_paths += [
    get_path(
        results_dir=None,
        learning_type='dl',
        algorithm_or_model_name='DeepConvNet',
        epoching=ival,
        fold_type='transfer_learning_frozen',
        n_folds=x,
        deprecated=False
    )
    for x in frozen_layers_list
]

# getting file_path
csv_paths = [
    os.path.join(x, 'statistics', 'tables', 'acc.csv')
    for x in folder_paths
]

"""
SETTING PATHS
"""
# saving stuff
savings_dir = join(dirname(
    dirname(dirname(folder_paths[1]))), 'learning_curve')
touch_dir(savings_dir)

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
    temp_data = []
    for csv_line in csv:
        temp_data.append(csv_line[-2])
    subj_data.append(temp_data)
    mean_data.append(np.mean(temp_data))
    stdd_data.append(np.std(temp_data))

"""
DEMIXING STUFF
"""
pos_layers_m = np.round(np.array(mean_data[0:6] + [mean_data[-1]]) * 100, 1)
neg_layers_m = np.round(np.array([mean_data[0]] + mean_data[-6:]) * 100, 1)
pos_layers_s = np.round(np.array(stdd_data[0:6] + [stdd_data[-1]]) * 100, 1)
neg_layers_s = np.round(np.array([stdd_data[0]] + stdd_data[-6:]) * 100, 1)

"""
PLOTTING POS
"""
plt.figure(dpi=100, figsize=fig_size, facecolor='w', edgecolor='k')
plt.style.use('seaborn-whitegrid')
plt.errorbar(x=[0, 1, 2, 3, 4, 5, 6],
             y=pos_layers_m,
             yerr=pos_layers_s,
             fmt='-.o', color='b', ecolor='r',
             linewidth=2, elinewidth=3, capsize=20, capthick=2)
plt.xlabel('indice di congelamento (da nessuno strato congelato a tutti)',
           fontsize=fontsize_1)
plt.ylabel('accuratezza (%)', fontsize=fontsize_1)
plt.xticks(fontsize=fontsize_2)
plt.yticks(fontsize=fontsize_2)
plt.title('esempi di training: 128', fontsize=fontsize_1)
savefig(
    join(savings_dir, 'frozen_layers_learning_curve_1'), bbox_inches='tight')

"""
PLOTTING NEG
"""
plt.figure(dpi=100, figsize=fig_size, facecolor='w', edgecolor='k')
plt.style.use('seaborn-whitegrid')
plt.errorbar(x=[0, -1, -2, -3, -4, -5, -6],
             y=neg_layers_m,
             yerr=neg_layers_s,
             fmt='-.o', color='b', ecolor='r',
             linewidth=2, elinewidth=3, capsize=20, capthick=2)
plt.xlabel('indice di congelamento (da tutti gli strati congelati a nessuno)',
           fontsize=fontsize_1)
plt.ylabel('accuratezza (%)', fontsize=fontsize_1)
plt.xticks(fontsize=fontsize_2)
plt.yticks(fontsize=fontsize_2)
plt.title('esempi di training: 128', fontsize=fontsize_1)
savefig(
    join(savings_dir, 'frozen_layers_learning_curve_2'), bbox_inches='tight')
