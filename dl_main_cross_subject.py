from os import getcwd
from os.path import join
from os.path import dirname
from collections import OrderedDict
from numpy.random import RandomState
from hgdecode.utils import create_log
from hgdecode.utils import print_manager
from hgdecode.loaders import CrossSubject
from hgdecode.classes import CrossValidation
from hgdecode.experiments import DLExperiment
from keras import backend as K

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

# setting random_state
random_state = RandomState(1234)

# setting subject_ids
subject_ids = (1, 2)  # , 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)

# setting hyperparameters
ival = (-500, 4000)
standardize_mode = 2
learning_rate = 1 * 1e-4
dropout_rate = 0.5
batch_size = 32
epochs = 800

"""
STARTING LOADING ROUTINE & COMPUTATION
Here you can change some parameter in function calls as well
"""
# creating a log object
subj_results_dir = create_log(
    results_dir=results_dir,
    learning_type='dl',
    algorithm_or_model_name=model_name,
    subject_id='subj_cross',
    output_on_file=False
)

# creating a cross-subject object for cross-subject validation
cross_obj = CrossSubject(data_dir=data_dir,
                         subject_ids=subject_ids,
                         channel_names=channel_names,
                         name_to_start_codes=name_to_start_codes,
                         random_state=random_state,
                         validation_frac=0.1,
                         resampling_freq=250,
                         train_test_split=True,
                         clean_ival_ms=ival,
                         epoch_ival_ms=ival,
                         clean_on_all_channels=False)

# parsing all cnt data to epoched (we no more need cnt)
cross_obj.parser(output_format='epo', parsing_type=1)

# pre-allocating experiment
exp = None

# cycling on subject leaved apart
for leave_subj in subject_ids:
    # clearing TF graph (https://github.com/keras-team/keras/issues/3579)
    print_manager('CLEARING KERAS BACKEND', print_style='double-dashed')
    K.clear_session()
    print_manager(print_style='last', bottom_return=1)

    # creating dataset for this "all but" fold
    cross_obj.parser(output_format='EEGDataset',
                     leave_subj=leave_subj,
                     parsing_type=1)

    # creating experiment instance
    exp = DLExperiment(
        # non-default inputs
        dataset=cross_obj.fold_data,
        model_name=model_name,
        results_dir=results_dir,
        name_to_start_codes=name_to_start_codes,
        random_state=random_state,
        fold_idx=leave_subj - 1,

        # hyperparameters
        dropout_rate=dropout_rate,  # Schirrmeister: 0.5
        learning_rate=learning_rate,  # Schirrmeister: ?
        batch_size=batch_size,  # Schirrmeister: 512
        epochs=epochs,  # Schirrmeister: ?
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
        save_model_at_each_epoch=False,
        subj_results_dir=subj_results_dir
    )

    # running training
    exp.train()

# at the very end, running cross-validation
if exp is not None:
    CrossValidation.cross_validate(subj_results_dir=exp.subj_results_dir,
                                   label_names=name_to_start_codes)
