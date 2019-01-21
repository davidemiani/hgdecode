from os import getcwd
from os.path import join
from os.path import dirname
from collections import OrderedDict
from numpy.random import RandomState
from hgdecode.utils import create_log
from hgdecode.loaders import CrossSubject
from hgdecode.classes import CrossValidation
from hgdecode.experiments import DLExperiment

# %%
"""
SETTING PARAMETERS
Here you can set whatever parameter you want
"""
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

# setting random seed
random_seed = RandomState(1234)

# setting subject_ids
subject_ids = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)

# recovery information
restart_from = None

# %%
"""
STARTING LOADING ROUTINE & COMPUTATION
Here you can change some parameter in function calls as well
"""
if restart_from is None:
    use_last_result_directory = False
else:
    use_last_result_directory = True

# creating a log object
create_log(
    results_dir=results_dir,
    learning_type='dl',
    algorithm_or_model_name=model_name,
    subject_id='subj_cross',
    output_on_file=False,
    use_last_result_directory=use_last_result_directory
)

# creating a cross-subject object for cross-subject validation
cross_obj = CrossSubject(data_dir=data_dir,
                         subject_ids=subject_ids,
                         channel_names=channel_names,
                         name_to_start_codes=name_to_start_codes,
                         resampling_freq=250,
                         train_test_split=True,
                         clean_ival_ms=(-1000, 1000),
                         epoch_ival_ms=(-1000, 1000),
                         clean_on_all_channels=False)

# if computation crashed, you can restart using only a subject subset
if restart_from is not None:
    subject_ids = restart_from

# pre-allocating experiment
exp = None

# cycling on subject leaved apart
for leave_subj in subject_ids:
    cross_obj.parser(output_format='EEGDataset',
                     leave_subj=leave_subj,
                     validation_frac=0.1,
                     parsing_type=1)

    # creating experiment instance
    exp = DLExperiment(
        # non-default inputs
        dataset=cross_obj.fold_data,
        model_name=model_name,
        results_dir=results_dir,
        name_to_start_codes=name_to_start_codes,
        random_seed=random_seed,
        fold_idx=leave_subj - 1,

        # hyperparameters
        dropout_rate=0.5,  # Schirrmeister: 0.5
        learning_rate=1 * 1e-4,  # Schirrmeister: ?
        batch_size=64,  # Schirrmeister: 512
        epochs=600,  # Schirrmeister: ?
        early_stopping=False,  # Schirrmeister: ?
        monitor='val_acc',  # Schirrmeister: ?
        min_delta=0.0001,  # Schirrmeister: ?
        patience=5,  # Schirrmeister: ?
        loss='categorical_crossentropy',  # Schirrmeister: ad hoc
        optimizer='Adam',  # Schirrmeister: Adam
        shuffle=True,  # Schirrmeister: ?
        crop_sample_size=None,  # Schirrmeister: 1125
        crop_step=None,  # Schirrmeister: 1

        # other parameters
        subject_id='_cross',
        data_generator=False,  # Schirrmeister: True
        save_model_at_each_epoch=False
    )

    # running training
    exp.train()

# at the very end, running cross-validation
if exp is not None:
    CrossValidation.cross_validate(subj_results_dir=exp.subj_results_dir,
                                   label_names=name_to_start_codes)
