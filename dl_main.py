from os import getcwd
from os.path import join
from os.path import dirname
from collections import OrderedDict
from numpy.random import RandomState
from hgdecode.utils import create_log
from hgdecode.loaders import dl_loader
from hgdecode.classes import CrossValidation
from hgdecode.experiments import DLExperiment

"""
SETTING PARAMETERS
------------------
In the following, you have to set / modify all the parameters to use for
further computation.

Parameters
----------
channel_names : list
    Channels to use for computation
data_dir : str
    Path to the directory that contains dataset
model_name : str
    Name of the Deep Learning model
name_to_start_codes : OrderedDict
    All possible classes names and codes in an ordered dict format
random_seed : rng seed
    Seed random for all random calls
results_dir : str
    Path to the directory that will contain the results
subject_ids : tuple
    All the subject ids in a tuple; add or remove subjects to run the
    algorithm for them or not
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

# setting random seed
random_seed = RandomState(1234)

"""
MAIN CYCLE
----------
For each subject, a new log will be created and the specific dataset loaded;
this dataset will be used to create an instance of the experiment; then the
experiment will be run. You can of course change all the experiment inputs
to obtain different results.
"""
for subject_id in subject_ids:
    # creating a log object
    create_log(
        results_dir=results_dir,
        learning_type='dl',
        algorithm_or_model_name=model_name,
        subject_id=subject_id,
        output_on_file=False
    )

    # loading epoched signal
    epo = dl_loader(
        data_dir=data_dir,
        name_to_start_codes=name_to_start_codes,
        channel_names=channel_names,
        subject_id=subject_id,
        resampling_freq=250,  # Schirrmeister: 250
        clean_ival_ms=(0, 4000),  # Schirrmeister: (0, 4000)
        epoch_ival_ms=(-500, 4000),  # Schirrmeister: (-500, 4000)
        train_test_split=True,  # Schirrmeister: True
        clean_on_all_channels=False  # Schirrmeister: True
    )

    # creating CrossCorrelation class instance
    cross_validation = CrossValidation(
        epo=epo,
        n_folds=8,
        validation_frac=0.1,
        random_seed=random_seed,
        shuffle=True
    )

    # pre-allocating experiment
    exp = None

    # cycling on folds for cross validation
    for fold_idx, current_fold in enumerate(cross_validation.folds):
        # creating EEGDataset for current fold
        dataset = cross_validation.create_dataset_for_fold(
            epo=epo,
            fold=current_fold
        )

        # creating experiment instance
        exp = DLExperiment(
            # non-default inputs
            dataset=dataset,
            model_name=model_name,
            results_dir=results_dir,
            name_to_start_codes=name_to_start_codes,
            random_seed=random_seed,
            fold_idx=fold_idx,

            # hyperparameters
            dropout_rate=0.5,  # Schirrmeister: 0.5
            learning_rate=0.001,  # Schirrmeister: ?, still not supported
            batch_size=32,  # Schirrmeister: 512
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

        # training
        exp.train()

    if exp is not None:
        # computing cross-validation
        cross_validation.cross_validate(
            subj_results_dir=exp.subj_results_dir,
            figures_dir=exp.figures_dir,
            tables_dir=exp.tables_dir
        )
