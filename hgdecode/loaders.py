import logging as log
from copy import deepcopy
from numpy import max
from numpy import abs
from numpy import sum
from numpy import mean
from os.path import join
from hgdecode.utils import print_manager
from braindecode.datasets.bbci import BBCIDataset
from braindecode.mne_ext.signalproc import mne_apply
from braindecode.mne_ext.signalproc import resample_cnt
from braindecode.mne_ext.signalproc import concatenate_raws_with_events
from braindecode.datautil.signalproc import bandpass_cnt
from braindecode.datautil.trial_segment import \
    create_signal_target_from_raw_mne


# %% GET_DATA_FILES_PATHS
def get_data_files_paths(data_dir, subject_id=1, train_test_split=True):
    # compute file name (for both train and test path)
    file_name = '{:d}.mat'.format(subject_id)

    # compute file paths
    if train_test_split:
        train_file_path = join(data_dir, 'train', file_name)
        test_file_path = join(data_dir, 'test', file_name)
        file_path = [train_file_path, test_file_path]
    else:
        file_path = [join(data_dir, 'train', file_name)]

    # return paths
    return file_path


# %% LOAD_CNT
def load_cnt(file_path, channel_names, clean_on_all_channels=True):
    # if we have to run the cleaning procedure on all channels, putting
    # load_sensor_names to None will assure us the BBCIDataset class will
    # load all possible sensors
    if clean_on_all_channels is True:
        channel_names = None

    # create the loader object for BBCI standard
    loader = BBCIDataset(file_path, load_sensor_names=channel_names)

    # load data
    return loader.load()


# %% GET_CLEAN_TRIAL_MASK
def get_clean_trial_mask(cnt, name_to_start_codes, clean_ival_ms=(0, 4000)):
    """
    Scan trial in continuous data and create a mask with only the
    valid ones; in this way, at the and of the loading routine,
    after all the data pre-processing, you will be able to cut away
    the original not valid data.
    """
    # split cnt into trials data for cleaning
    set_for_cleaning = create_signal_target_from_raw_mne(
        cnt,
        name_to_start_codes,
        clean_ival_ms
    )

    # compute the clean_trial_mask: in this case we take only all
    # trials that have absolute microvolt values larger than +- 800
    clean_trial_mask = max(abs(set_for_cleaning.X), axis=(1, 2)) < 800

    # logging clean trials information
    log.info(
        'Clean trials: {:3d}  of {:3d} ({:5.1f}%)'.format(
            sum(clean_trial_mask),
            len(set_for_cleaning.X),
            mean(clean_trial_mask) * 100)
    )

    # return the clean_trial_mask
    return clean_trial_mask


# %% PICK_RIGHT_CHANNELS
def pick_right_channels(cnt, channel_names):
    # return the same cnt but with only right channels
    return cnt.pick_channels(channel_names)


# %% STANDARDIZE_CNT
def standardize_cnt(cnt):
    # normalize data
    cnt = mne_apply(
        lambda x: x - mean(x, axis=0, keepdims=True),
        cnt
    )

    # computing frequencies
    sampling_freq = cnt.info['sfreq']
    init_freq = 0.1
    stop_freq = sampling_freq / 2 - 0.1

    # cut away DC and too high frequencies
    cnt = mne_apply(
        lambda x: bandpass_cnt(x, init_freq, stop_freq, sampling_freq),
        cnt
    )
    return cnt


# %% LOAD_AND_PREPROCESS_DATA
def load_and_preprocess_data(data_dir,
                             name_to_start_codes,
                             channel_names,
                             subject_id=1,
                             resampling_freq=None,
                             clean_ival_ms=(0, 4000),
                             train_test_split=True,
                             clean_on_all_channels=True):
    # TODO: create here another get_data_files_paths function if you have a
    #  different file configuration; in every case, file_paths must be a
    #  list of paths to valid BBCI standard files
    # getting data paths
    file_paths = get_data_files_paths(
        data_dir,
        subject_id=subject_id,
        train_test_split=train_test_split
    )

    # starting the loading routine
    print_manager('DATA LOADING ROUTINE', 'double-dashed')
    print_manager('Loading continuous data...')

    # pre-allocating main cnt
    cnt = None

    # loading files and merging them
    for idx, current_path in enumerate(file_paths):
        current_cnt = load_cnt(file_path=current_path,
                               channel_names=channel_names,
                               clean_on_all_channels=clean_on_all_channels)
        # if the path is the first one...
        if idx is 0:
            # ...copying current_cnt as the main one, else...
            cnt = deepcopy(current_cnt)
        else:
            # merging current_cnt with the main one
            cnt = concatenate_raws_with_events([cnt, current_cnt])
    print_manager('DONE!!', bottom_return=1)

    # getting clean_trial_mask
    print_manager('Getting clean trial mask...')
    clean_trial_mask = get_clean_trial_mask(
        cnt=cnt,
        name_to_start_codes=name_to_start_codes,
        clean_ival_ms=clean_ival_ms
    )
    print_manager('DONE!!', bottom_return=1)

    # pick only right channels
    log.info('Picking only right channels...')
    cnt = pick_right_channels(cnt, channel_names)
    print_manager('DONE!!', bottom_return=1)

    # resample continuous data
    if resampling_freq is not None:
        log.info('Resampling continuous data...')
        cnt = resample_cnt(
            cnt,
            resampling_freq
        )
        print_manager('DONE!!', bottom_return=1)

    # standardize continuous data
    log.info('Standardizing continuous data...')
    cnt = standardize_cnt(cnt)
    print_manager('DONE!!', 'last', bottom_return=1)

    return cnt, clean_trial_mask


# %% ML_LOADER
def ml_loader(data_dir,
              name_to_start_codes,
              channel_names,
              subject_id=1,
              resampling_freq=None,
              clean_ival_ms=(0, 4000),
              train_test_split=True,
              clean_on_all_channels=True):
    outputs = load_and_preprocess_data(
        data_dir=data_dir,
        name_to_start_codes=name_to_start_codes,
        channel_names=channel_names,
        subject_id=subject_id,
        resampling_freq=resampling_freq,
        clean_ival_ms=clean_ival_ms,
        train_test_split=train_test_split,
        clean_on_all_channels=clean_on_all_channels
    )
    return outputs[0], outputs[1]


# %% DL_LOADER
def dl_loader(data_dir,
              name_to_start_codes,
              channel_names,
              subject_id=1,
              resampling_freq=None,
              clean_ival_ms=(0, 4000),
              epoch_ival_ms=(-500, 4000),
              train_test_split=True,
              clean_on_all_channels=True):
    # loading and pre-processing data
    cnt, clean_trial_mask = load_and_preprocess_data(
        data_dir=data_dir,
        name_to_start_codes=name_to_start_codes,
        channel_names=channel_names,
        subject_id=subject_id,
        resampling_freq=resampling_freq,
        clean_ival_ms=clean_ival_ms,
        train_test_split=train_test_split,
        clean_on_all_channels=clean_on_all_channels
    )
    print_manager('EPOCHING AND CLEANING WITH MASK', 'double-dashed')

    # epoching continuous data (from RawArray to SignalAndTarget)
    print_manager('Epoching...')
    epo = create_signal_target_from_raw_mne(
        cnt,
        name_to_start_codes,
        epoch_ival_ms
    )
    print_manager('DONE!!', bottom_return=1)

    # cleaning epoched signal with mask
    print_manager('cleaning with mask...')
    epo.X = epo.X[clean_trial_mask]
    epo.y = epo.y[clean_trial_mask]
    print_manager('DONE!!', 'last', bottom_return=1)

    # returning only the epoched signal
    return epo
