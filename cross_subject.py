from os import getcwd
from os.path import join
from os.path import dirname
from collections import OrderedDict
from numpy.random import RandomState
from hgdecode.utils import create_log
from hgdecode.loaders import CrossSubject

# setting model_name
model_name = 'DeepConvNet'

# setting channel_names
channel_names = ['FC5', 'FC1', 'FC2', 'FC6', 'C3', 'C4',
                 'CP5', 'CP1', 'CP2', 'CP6',
                 'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2', 'C6',
                 'CP3', 'CPz', 'CP4',
                 'FFC5h', 'FFC3h', 'FFC4h', 'FFC6h',
                 'FCC5h', 'FCC3h', 'FCC4h', 'FCC6h',
                 'CCP5h', 'CCP3h', 'CCP4h', 'CCP6h',
                 'CPP5h', 'CPP3h', 'CPP4h', 'CPP6h',
                 'FFC1h', 'FFC2h', 'FCC1h', 'FCC2h',
                 'CCP1h', 'CCP2h', 'CPP1h', 'CPP2h']

# setting data_dir & results_dir
data_dir = join(dirname(dirname(getcwd())), 'datasets', 'High-Gamma')
results_dir = join(dirname(dirname(getcwd())), 'results', 'hgdecode')

# setting name_to_start_codes
name_to_start_codes = OrderedDict([('Right Hand', [1]),
                                   ('Left Hand', [2]),
                                   ('Rest', [3]),
                                   ('Feet', [4])])

# setting subject_ids
subject_ids = (1, 2,)  # 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)

# setting random seed
random_seed = RandomState(1234)

# creating a log object
create_log(
    results_dir=results_dir,
    learning_type='dl',
    algorithm_or_model_name=model_name,
    subject_id='cross_subj',
    output_on_file=False
)

cross_obj = CrossSubject(data_dir=data_dir,
                         subject_ids=subject_ids,
                         channel_names=channel_names,
                         name_to_start_codes=name_to_start_codes,
                         resampling_freq=250,
                         train_test_split=True,
                         clean_ival_ms=(-500, 4000),
                         epoch_ival_ms=(-500, 4000),
                         clean_on_all_channels=False)

for leave_subj in subject_ids:
    cross_obj.parser('EEGDataset', leave_subj, validation_frac=0.1)
    print('ciao!')
