from os import getcwd
from os.path import join
from os.path import dirname
from collections import OrderedDict
from hgdecode.utils import create_log
from hgdecode.utils import print_manager
from hgdecode.loaders import ml_loader
from hgdecode.experiments import FBCSPrLDAExperiment

"""
SETTING PARAMETERS
------------------
In the following, you have to set / modify all the parameters to use for 
further computation.
    
Parameters
----------
algorithm_name : str
    Machine Learning algorithm name that is going to be used
channel_names : list
    Channels to use for computation
data_dir : str
    Path to the directory that contains dataset
name_to_start_codes : OrderedDict
    All possible classes names and codes in an ordered dict format
results_dir : str
    
subject_ids : tuple
    All the subject ids in a tuple; add or remove subjects to run the
    algorithm for them or not
"""
# setting ml_algorithm
algorithm_name = 'FBCSP_rLDA'

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
        learning_type='ml',
        algorithm_or_model_name=algorithm_name,
        subject_id=subject_id,
        output_on_file=False
    )

    # loading dataset
    cnt, clean_trial_mask = ml_loader(
        data_dir=data_dir,
        name_to_start_codes=name_to_start_codes,
        channel_names=channel_names,
        subject_id=subject_id,
        resampling_freq=250,
        clean_ival_ms=[0, 4000]
    )

    # creating experiment instance
    exp = FBCSPrLDAExperiment(
        # signal-related inputs
        cnt,
        clean_trial_mask,
        name_to_start_codes,
        name_to_stop_codes=None,
        epoch_ival_ms=(-500, 4000),

        # bank filter-related inputs
        min_freq=[0, 10],
        max_freq=[12, 122],
        window=[6, 8],
        overlap=[3, 4],
        filt_order=3,

        # machine learning parameters
        n_folds=10,
        n_top_bottom_csp_filters=5,
        n_selected_filterbands=None,
        n_selected_features=20,
        forward_steps=2,
        backward_steps=1,
        stop_when_no_improvement=False,
        shuffle=False,
        average_trial_covariance=True
    )

    # running the experiment
    exp.run()
