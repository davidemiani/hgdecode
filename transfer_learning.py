from os import getcwd
from numpy import ceil
from os.path import join
from os.path import dirname
from collections import OrderedDict
from numpy.random import RandomState
from hgdecode.utils import get_path
from hgdecode.utils import create_log
from hgdecode.utils import print_manager
from hgdecode.utils import clear_all_models
from hgdecode.loaders import dl_loader
from hgdecode.classes import CrossValidation
from hgdecode.experiments import DLExperiment
from keras import backend as K

"""
SETTING PARAMETERS
------------------
"""

# setting model_name and validation_frac
model_name = 'DeepConvNet'  # Schirrmeister: 'DeepConvNet' or 'ShallowNet'

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
subject_ids = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)

# setting random_state
random_state = RandomState(1234)

# setting fold_size: this will be the number of trials for training,
# so it must be multiple of 4
fold_size = 8  # must be integer
validation_frac = 0.1

# setting frozen_layers
layers_to_freeze = 0  # can be between -6 and 6 for DeepConvNet

# other hyper-parameters
dropout_rate = 0.6
learning_rate = 2 * 1e-5
epochs = 100
ival = (-1000, 1000)

"""
GETTING CROSS-SUBJECT MODELS DIR PATH
-------------------------------------
"""
# setting cross_subj_dir_path: data from cross-subj computation are stored here
learning_type = 'dl'
algorithm_or_model_name = None
epoching = ival
fold_type = 'cross_subject'
n_folds = None
deprecated = False
cross_subj_dir_path = get_path(
    results_dir=dirname(results_dir),
    learning_type=learning_type,
    algorithm_or_model_name=algorithm_or_model_name,
    epoching=epoching,
    fold_type=fold_type,
    n_folds=n_folds,
    deprecated=deprecated
)

"""
COMPUTATION
-----------
"""
for subject_id in subject_ids:
    # creating a log object
    subj_results_dir = create_log(
        results_dir=results_dir,
        learning_type='dl',
        algorithm_or_model_name=model_name,
        subject_id=subject_id,
        output_on_file=False,
        use_last_result_directory=False
    )

    # loading epoched signal
    epo = dl_loader(
        data_dir=data_dir,
        name_to_start_codes=name_to_start_codes,
        channel_names=channel_names,
        subject_id=subject_id,
        resampling_freq=250,  # Schirrmeister: 250
        clean_ival_ms=ival,  # Schirrmeister: (0, 4000)
        epoch_ival_ms=ival,  # Schirrmeister: (-500, 4000)
        train_test_split=True,  # Schirrmeister: True
        clean_on_all_channels=False  # Schirrmeister: True
    )

    # if fold_size is not a multiple of 4, putting it to the nearest
    fold_size = int(ceil(fold_size / 4) * 4)

    # computing batch_size to be...
    if fold_size <= 64:
        batch_size = fold_size
    else:
        batch_size = 64

    # I don't think this is a good idea:
    # validation_size is equal to fold_size
    # validation_size = fold_size

    # creating CrossValidation class instance
    cross_validation = CrossValidation(
        X=epo.X,
        y=epo.y,
        fold_size=fold_size,
        # validation_size=validation_size,
        validation_frac=validation_frac,
        random_state=random_state, shuffle=True,
        swap_train_test=True,
    )
    cross_validation.balance_train_set(train_size=fold_size)

    # pre-allocating experiment
    exp = None

    # cycling on folds for cross validation
    for fold_idx, current_fold in enumerate(cross_validation.folds):
        # clearing TF graph (https://github.com/keras-team/keras/issues/3579)
        print_manager('CLEARING KERAS BACKEND', print_style='double-dashed')
        K.clear_session()
        print_manager(print_style='last', bottom_return=1)

        # printing fold information
        print_manager(
            'SUBJECT {}, FOLD {}'.format(subject_id, fold_idx + 1),
            print_style='double-dashed'
        )
        cross_validation.print_fold_classes(fold_idx)
        print_manager(print_style='last', bottom_return=1)

        # creating EEGDataset for current fold
        dataset = cross_validation.create_dataset(fold=current_fold)

        # creating experiment instance
        exp = DLExperiment(
            # non-default inputs
            dataset=dataset,
            model_name=model_name,
            results_dir=results_dir,
            subj_results_dir=subj_results_dir,
            name_to_start_codes=name_to_start_codes,
            random_state=random_state,
            fold_idx=fold_idx,

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
            subject_id=subject_id,
            data_generator=False,  # Schirrmeister: True
            save_model_at_each_epoch=False
        )

        # loading model weights from cross-subject pre-trained model
        exp.prepare_for_transfer_learning(
            cross_subj_dir_path=cross_subj_dir_path,
            subject_id=subject_id,
            train_anyway=True
        )

        # freezing layers
        exp.freeze_layers(layers_to_freeze=layers_to_freeze)

        # training
        exp.train()

    # computing cross-validation
    if exp is not None:
        cross_validation.cross_validate(
            subj_results_dir=exp.subj_results_dir,
            label_names=name_to_start_codes)

    # clearing all models (they are not useful once we have results)
    clear_all_models(subj_results_dir)
