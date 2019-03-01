import numpy as np
import matplotlib.pyplot as plt
from pylab import savefig
from pickle import load
from os.path import join
from os.path import dirname
from hgdecode.utils import listdir2
from hgdecode.utils import touch_dir
from hgdecode.utils import get_subj_str
from hgdecode.utils import get_fold_str
from hgdecode.utils import get_path


def get_fold_number(folder_path, subj_str):
    return len(listdir2(join(folder_path, subj_str)))


"""
SET HERE YOUR PARAMETERS
"""
# to find file parameters
results_dir = None
learning_type = 'dl'
algorithm_or_model_name = None
epoching = '-1000_1000'
fold_type = 'transfer_learning'
train_size_list = [4, 8, 16, 32, 64, 128]  # must be a list of integer
deprecated = False

fontsize_1 = 35
fontsize_2 = 27.5
fig_size = (22, 7.5)

"""
GETTING PATHS
"""
# trainings stuff
folder_paths = [
    get_path(
        results_dir=results_dir,
        learning_type=learning_type,
        algorithm_or_model_name=algorithm_or_model_name,
        epoching=epoching,
        fold_type=fold_type,
        n_folds=x,
        deprecated=deprecated
    )
    for x in train_size_list
]
n_trainings = len(folder_paths)

# saving stuff
savings_dir = join(dirname(dirname(folder_paths[0])), 'learning_curve')
touch_dir(savings_dir)

"""
SUBJECTS STUFF
"""
# subject stuff
n_trials_list = [480, 973, 1040, 1057, 880, 1040, 1040, 814, 1040, 1040,
                 1040, 1040, 950, 1040]
n_subjects = len(n_trials_list)
subj_str_list = [get_subj_str(x + 1) for x in range(n_subjects)]

"""
COMPUTATION STARTS HERE
"""
# pre-allocating results dictionary
results = {
    'n_folds': [],
    'n_trials': [],
    'n_train_trials': [],
    'n_valid_trials': [],
    'n_test_trials': [],
    'perc_train_trials': [],
    'perc_valid_trials': [],
    'perc_test_trials': [],
    'm_acc': [],
    's_acc': []
}

# cycling on subject
for subj, current_n_trials in zip(subj_str_list, n_trials_list):
    n_folds = []
    n_trials = []
    n_train_trials = []
    n_valid_trials = []
    n_test_trials = []
    perc_train_trials = []
    perc_valid_trials = []
    perc_test_trials = []
    m_acc = []
    s_acc = []

    # cycling on all possible fold splits
    for idx, current_train_size in enumerate(train_size_list):
        n_folds.append(get_fold_number(folder_paths[idx], subj))
        n_trials.append(current_n_trials)
        n_valid_trials.append(int(np.floor(n_trials[idx] * 0.1)))
        n_train_trials.append(current_train_size)
        n_test_trials.append(n_trials[idx] - n_train_trials[idx] -
                             n_valid_trials[idx])
        perc_train_trials.append(
            np.round(n_train_trials[idx] / n_trials[idx] * 100, 1))
        perc_valid_trials.append(
            np.round(n_valid_trials[idx] / n_trials[idx] * 100, 1))
        perc_test_trials.append(
            np.round(100 - perc_valid_trials[idx] - perc_train_trials[idx], 1))

        # cycling on folds
        folds_acc = []
        for fold_str in [get_fold_str(x + 1) for x in range(n_folds[idx])]:
            file_path = join(folder_paths[idx],
                             subj, fold_str, 'fold_stats.pickle')
            with open(file_path, 'rb') as f:
                fold_stats = load(f)
            folds_acc.append(fold_stats['test']['acc'])
        m_acc.append(np.mean(folds_acc) * 100)
        s_acc.append(np.std(folds_acc) * 100)

    # assigning results for this subject
    results['n_folds'].append(n_folds)
    results['n_trials'].append(n_trials)
    results['n_train_trials'].append(n_train_trials)
    results['n_valid_trials'].append(n_valid_trials)
    results['n_test_trials'].append(n_test_trials)
    results['perc_train_trials'].append(perc_train_trials)
    results['perc_valid_trials'].append(perc_valid_trials)
    results['perc_test_trials'].append(perc_test_trials)
    results['m_acc'].append(m_acc)
    results['s_acc'].append(s_acc)

    # plotting learning curve for this subject
    m_acc = np.array(m_acc)
    s_acc = np.array(s_acc)
    plot_path = join(savings_dir, subj)
    if learning_type is 'dl':
        title = '{} transfer learning curve\n'.format(subj) + \
                '({} samples, {} validation samples)'.format(
                    n_trials[0], n_valid_trials[0])
    else:
        title = '{} transfer learning curve\n({} samples)'.format(subj,
                                                                  n_trials[0])
    x_tick_labels = ['{}\n({} folds)'.format(trials, folds)
                     for trials, folds in zip(n_train_trials, n_folds)]
    plt.figure(dpi=100, figsize=(12.8, 7.2), facecolor='w', edgecolor='k')
    plt.style.use('seaborn-whitegrid')
    plt.errorbar(x=[2, 4, 6, 8, 10, 12], y=m_acc, yerr=s_acc,
                 fmt='-.o', color='b', ecolor='r',
                 linewidth=2, elinewidth=3, capsize=20, capthick=2)
    plt.xlabel('training samples', fontsize=25)
    plt.ylabel('accuracy (%)', fontsize=25)
    plt.xticks([2, 4, 6, 8, 10, 12], labels=x_tick_labels, fontsize=20)
    plt.yticks(fontsize=20)
    plt.title(title, fontsize=25)

    # saving figure
    savefig(plot_path, bbox_inches='tight')

# getting data for last plot
n_trials = np.array(results['n_trials'])
n_train_trials = np.array(results['n_train_trials'])
n_valid_trials = np.array(results['n_valid_trials'])
n_folds = np.array(results['n_folds'])

# averaging data
m_n_trials = int(np.round(np.mean(n_trials, axis=0))[0].tolist())
s_n_trials = int(np.round(np.std(n_trials, axis=0))[0].tolist())
m_n_train_trials = train_size_list
m_n_valid_trials = np.round(np.mean(n_valid_trials, axis=0)).tolist()
s_n_valid_trials = np.round(np.std(n_valid_trials, axis=0)).tolist()
m_n_valid_trials = list(map(int, m_n_valid_trials))
s_n_valid_trials = list(map(int, s_n_valid_trials))
m_n_folds = np.round(np.mean(n_folds, axis=0)).tolist()
s_n_folds = np.round(np.std(n_folds, axis=0)).tolist()
m_n_folds = list(map(int, m_n_folds))
s_n_folds = list(map(int, s_n_folds))
m_acc = np.mean(np.array(results['m_acc']), axis=0)
s_acc = np.mean(np.array(results['s_acc']), axis=0)

# plotting learning curve for total mean
plot_path = join(savings_dir, 'transfer_learning_curve')
if learning_type is 'dl':
    # title = '{} learning curve\n({}$\pm${} samples, {}$\pm${} validation ' \
    #         'samples)'.format('average', m_n_trials, s_n_trials,
    #                           m_n_valid_trials[0], s_n_valid_trials[0])
    title = 'totale esempi per soggetto: {}$\pm${}; esempi di validazione: ' \
            '{}$\pm${}'.format(m_n_trials, s_n_trials,
                               m_n_valid_trials[0], s_n_valid_trials[0])
else:
    title = '{} learning curve\n({}$\pm${} samples)'.format(
        'average', m_n_trials, s_n_trials)
x_tick_labels = ['{}\n({}$\pm${} fold)'.format(
    m_n_train_trials[idx], m_n_folds[idx], s_n_folds[idx])
    for idx in range(n_trainings)]
plt.figure(dpi=100, figsize=fig_size, facecolor='w', edgecolor='k')
plt.style.use('seaborn-whitegrid')
plt.errorbar(x=[2, 4, 6, 8, 10, 12], y=m_acc, yerr=s_acc,
             fmt='-.o', color='b', ecolor='r',
             linewidth=2, elinewidth=3, capsize=20, capthick=2)
plt.xlabel('esempi di training', fontsize=fontsize_1)
plt.ylabel('accuratezza (%)', fontsize=fontsize_1)
plt.xticks([2, 4, 6, 8, 10, 12], labels=x_tick_labels, fontsize=fontsize_2)
plt.yticks(fontsize=fontsize_2)
plt.title(title, fontsize=fontsize_1)

# in case of single_subject, -500,4000
if fold_type is 'single_subject':
    if epoching is '-500_4000':
        plt.ylim(80, 100)

# saving figure
savefig(plot_path, bbox_inches='tight')
