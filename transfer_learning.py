from os import getcwd
from numpy import ceil
from os.path import join
from os.path import dirname
from collections import OrderedDict
from numpy.random import RandomState
from hgdecode.utils import create_log
from hgdecode.utils import get_fold_str
from hgdecode.loaders import dl_loader
from hgdecode.classes import CrossValidation
from hgdecode.experiments import DLExperiment

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

# setting cross_subj_dir_name: data from cross-subj computation are stored here
cross_subj_dir_name = '2019-01-18_13-33-01'

# setting fold_size: this will be the number of trials for training,
# so it must be multiple of 4
fold_size = 4  # must be integer

"""
COMPUTATION
-----------
"""
for subject_id in subject_ids:
    # creating a log object
    create_log(
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
        clean_ival_ms=(-500, 4000),  # Schirrmeister: (0, 4000)
        epoch_ival_ms=(-500, 4000),  # Schirrmeister: (-500, 4000)
        train_test_split=True,  # Schirrmeister: True
        clean_on_all_channels=False  # Schirrmeister: True
    )

    # if fold_size is not a multiple of 4, putting it to the nearest
    fold_size = int(ceil(fold_size / 4) * 4)

    # computing batch_size to be...
    if fold_size <= 64:
        batch_size = int(fold_size / 2)
    else:
        batch_size = 32

    # validation_size is equal to fold_size
    validation_size = fold_size

    # creating CrossValidation class instance
    cross_validation = CrossValidation(
        X=epo.X,
        y=epo.y,
        fold_size=fold_size,
        validation_size=validation_size,
        random_state=random_state, shuffle=True,
        swap_train_test=True,
    )

    # pre-allocating experiment
    exp = None

    # cycling on folds for cross validation
    for fold_idx, current_fold in enumerate(cross_validation.folds):
        # creating EEGDataset for current fold
        dataset = cross_validation.create_dataset(fold=current_fold)

        # creating experiment instance
        exp = DLExperiment(
            # non-default inputs
            dataset=dataset,
            model_name=model_name,
            results_dir=results_dir,
            name_to_start_codes=name_to_start_codes,
            random_state=random_state,
            fold_idx=fold_idx,

            # hyperparameters
            dropout_rate=0.5,  # Schirrmeister: 0.5
            learning_rate=5 * 1e-6,  # Schirrmeister: ?
            batch_size=batch_size,  # Schirrmeister: 512
            epochs=2,  # Schirrmeister: ?
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
        exp.model.load_weights(join(results_dir,
                                    'dl',
                                    model_name,
                                    cross_subj_dir_name,
                                    'subj_cross',
                                    get_fold_str(subject_id),
                                    'net_best_val_loss.h5'
                                    ))

        # training
        exp.train()

    # computing cross-validation
    if exp is not None:
        cross_validation.cross_validate(
            subj_results_dir=exp.subj_results_dir,
            label_names=name_to_start_codes)
