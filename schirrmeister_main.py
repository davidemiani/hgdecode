from os import getcwd
from os.path import join
from os.path import dirname
from collections import OrderedDict
from numpy.random import RandomState
from numpy import array
from numpy import floor
from numpy import setdiff1d
from hgdecode.utils import create_log
from hgdecode.utils import print_manager
from hgdecode.loaders import ml_loader
from hgdecode.classes import CrossValidation
from hgdecode.experiments import DLExperiment
from hgdecode.experiments import FBCSPrLDAExperiment
from keras import backend as K
from braindecode.datautil.trial_segment import \
    create_signal_target_from_raw_mne
from hgdecode.utils import ml_results_saver

"""
ONLY PARAMETER YOU CAN CHOSE
----------------------------
"""
# set here what type of learning you want
learning_type = 'ml'

"""
SETTING OTHER PARAMETERS (YOU CANNOT MODIFY THAT)
-------------------------------------------------
"""
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

# subject list
subject_ids = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)

# computing algorithm_or_model_name and standardize_mode
if learning_type == 'ml':
    algorithm_or_model_name = 'FBCSP_rLDA'
    standardize_mode = 0
else:
    algorithm_or_model_name = 'DeepConvNet'
    standardize_mode = 2

"""
MAIN CYCLE
----------
"""
for subject_id in subject_ids:
    # creating a log object
    subj_results_dir = create_log(
        results_dir=results_dir,
        learning_type=learning_type,
        algorithm_or_model_name=algorithm_or_model_name,
        subject_id=subject_id,
        output_on_file=False
    )

    # loading cnt signal
    cnt, clean_trial_mask = ml_loader(
        data_dir=data_dir,
        name_to_start_codes=name_to_start_codes,
        channel_names=channel_names,
        subject_id=subject_id,
        resampling_freq=250,  # Schirrmeister: 250
        clean_ival_ms=(-500, 4000),  # Schirrmeister: (0, 4000)
        train_test_split=True,  # Schirrmeister: True
        clean_on_all_channels=False,  # Schirrmeister: True
        standardize_mode=standardize_mode  # Schirrmeister: 2
    )

    # splitting two algorithms
    if learning_type == 'ml':
        # creating experiment instance
        exp = FBCSPrLDAExperiment(
            # signal-related inputs
            cnt=cnt,
            clean_trial_mask=clean_trial_mask,
            name_to_start_codes=name_to_start_codes,
            random_state=random_state,
            name_to_stop_codes=None,  # Schirrmeister: None
            epoch_ival_ms=(-500, 4000),  # Schirrmeister: (-500, 4000)

            # bank filter-related inputs
            min_freq=[0, 10],  # Schirrmeister: [0, 10]
            max_freq=[12, 122],  # Schirrmeister: [12, 122]
            window=[6, 8],  # Schirrmeister: [6, 8]
            overlap=[3, 4],  # Schirrmeister: [3, 4]
            filt_order=3,  # filt_order: 3

            # machine learning parameters
            n_folds=0,  # Schirrmeister: ?
            n_top_bottom_csp_filters=5,  # Schirrmeister: 5
            n_selected_filterbands=None,  # Schirrmeister: None
            n_selected_features=20,  # Schirrmeister: 20
            forward_steps=2,  # Schirrmeister: 2
            backward_steps=1,  # Schirrmeister: 1
            stop_when_no_improvement=False,  # Schirrmeister: False
            shuffle=False,  # Schirrmeister: False
            average_trial_covariance=True  # Schirrmeister: True
        )

        # running the experiment
        exp.run()

        # saving results for this subject
        ml_results_saver(exp=exp, subj_results_dir=subj_results_dir)

        # computing statistics for this subject
        CrossValidation.cross_validate(subj_results_dir=subj_results_dir,
                                       label_names=name_to_start_codes)
    elif learning_type == 'dl':
        # creating schirrmeister fold
        all_idxs = array(range(len(clean_trial_mask)))
        folds = [
            {
                'train': all_idxs[:-160],
                'test': all_idxs[-160:]
            }
        ]
        folds[0]['train'] = folds[0]['train'][clean_trial_mask[:-160]]
        folds[0]['test'] = folds[0]['test'][clean_trial_mask[-160:]]

        # adding validation
        valid_idxs = array(range(int(floor(len(clean_trial_mask) * 0.1))))
        folds[0]['train'] = setdiff1d(folds[0]['train'], valid_idxs)
        folds[0]['valid'] = valid_idxs

        # parsing cnt to epoched data
        print_manager('Epoching...')
        epo = create_signal_target_from_raw_mne(cnt,
                                                name_to_start_codes,
                                                (-500, 4000))
        print_manager('DONE!!', bottom_return=1)

        # # cleaning epoched signal with mask
        # print_manager('cleaning with mask...')
        # epo.X = epo.X[clean_trial_mask]
        # epo.y = epo.y[clean_trial_mask]
        # print_manager('DONE!!', 'last', bottom_return=1)

        # creating cv instance
        cv = CrossValidation(X=epo.X, y=epo.y, shuffle=False)

        # creating EEGDataset for current fold
        dataset = cv.create_dataset(fold=folds[0])

        # clearing TF graph (https://github.com/keras-team/keras/issues/3579)
        print_manager('CLEARING KERAS BACKEND', print_style='double-dashed')
        K.clear_session()
        print_manager(print_style='last', bottom_return=1)

        # creating experiment instance
        exp = DLExperiment(
            # non-default inputs
            dataset=dataset,
            model_name=algorithm_or_model_name,
            results_dir=results_dir,
            subj_results_dir=subj_results_dir,
            name_to_start_codes=name_to_start_codes,
            random_state=random_state,
            fold_idx=0,

            # hyperparameters
            dropout_rate=0.5,  # Schirrmeister: 0.5
            learning_rate=1 * 1e-4,  # Schirrmeister: ?
            batch_size=32,  # Schirrmeister: 512
            epochs=1000,  # Schirrmeister: ?
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

        # training
        exp.train()

        # computing cross-validation
        CrossValidation.cross_validate(subj_results_dir=subj_results_dir,
                                       label_names=name_to_start_codes)
