from numpy import max
from numpy import abs
from numpy import sum
from numpy import mean
from numpy import round
from numpy import arange
from numpy import int as npint
from os.path import join
from logging import getLogger
from hgdecode.utils import print_manager
from hgdecode.classes import EEGDataset
from braindecode.datasets.bbci import BBCIDataset
from braindecode.mne_ext.signalproc import mne_apply
from braindecode.mne_ext.signalproc import resample_cnt
from braindecode.mne_ext.signalproc import concatenate_raws_with_events
from braindecode.datautil.signalproc import bandpass_cnt
from braindecode.datautil.trial_segment import \
    create_signal_target_from_raw_mne

log = getLogger(__name__)


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
def load_cnt(file_path):
    # create the loader object for BBCI standard
    loader = BBCIDataset(file_path, load_sensor_names=None)

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

    # log clean trials information
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


def standardize_cnt(cnt):
    # normalize data
    cnt = mne_apply(
        lambda x: x - mean(x, axis=0, keepdims=True),
        cnt
    )

    # cut away DC and too high frequencies
    cnt = mne_apply(
        lambda x: bandpass_cnt(x, 0.1, 122.0, cnt.info['sfreq']),
        cnt
    )

    return cnt


# %% LOAD_AND_PREPROCESS_DATA
def load_and_preprocess_data(data_dir,
                             name_to_start_codes,
                             channel_names,
                             subject_id=1,
                             resampling_freq=250,
                             clean_ival_ms=(0, 4000),
                             train_test_split=True):
    # getting data paths
    file_paths = get_data_files_paths(
        data_dir,
        subject_id=subject_id,
        train_test_split=train_test_split
    )

    # starting the loading routine
    print_manager('DATA LOADING ROUTINE', 'double-dashed')
    print_manager('Loading continuous data...')

    # if exists only one data file...
    if len(file_paths) == 1:
        # ...loading just it, else...
        cnt = load_cnt(file_paths[0])
        train_len = None
    elif len(file_paths) == 2:
        # loading train_cnt, test_cnt and merging them
        train_cnt = load_cnt(file_paths[0])
        train_len = len(train_cnt.info['events'])
        test_cnt = load_cnt(file_paths[1])
        cnt = concatenate_raws_with_events([train_cnt, test_cnt])
    else:
        raise Exception('something went wrong: check single/multiple file '
                        'loading routine.')
    print_manager('Done!!', bottom_return=1)

    # getting clean_trial_mask
    print_manager('Getting clean trial mask...')
    clean_trial_mask = get_clean_trial_mask(
        cnt,
        name_to_start_codes,
        clean_ival_ms
    )
    print_manager('DONE!!', bottom_return=1)

    # pick only right channels
    log.info('Picking only right channels...')
    cnt = pick_right_channels(cnt, channel_names)
    print_manager('DONE!!', bottom_return=1)

    # resample continuous data
    log.info('Resampling continuous data...')
    cnt = resample_cnt(
        cnt,
        resampling_freq
    )
    print_manager('DONE!!', bottom_return=1)

    # standardize continuous data
    log.info('Standardizing continuous data...')
    cnt = standardize_cnt(cnt)
    print_manager('DONE!!', bottom_return=1)

    return cnt, clean_trial_mask, train_len


# %% ML_LOADER
def ml_loader(data_dir,
              name_to_start_codes,
              channel_names,
              subject_id=1,
              resampling_freq=250,
              clean_ival_ms=(0, 4000),
              train_test_split=True):
    outputs = load_and_preprocess_data(
        data_dir=data_dir,
        name_to_start_codes=name_to_start_codes,
        channel_names=channel_names,
        subject_id=subject_id,
        resampling_freq=resampling_freq,
        clean_ival_ms=clean_ival_ms,
        train_test_split=train_test_split
    )
    return outputs[0], outputs[1]


# %% DL_LOADER
def dl_loader(data_dir,
              name_to_start_codes,
              channel_names,
              subject_id=1,
              resampling_freq=250,
              clean_ival_ms=(0, 4000),
              epoch_ival_ms=(-500, 4000),
              train_test_split=True,
              validation_frac=0.2):
    cnt, clean_trial_mask, train_len = load_and_preprocess_data(
        data_dir=data_dir,
        name_to_start_codes=name_to_start_codes,
        channel_names=channel_names,
        subject_id=subject_id,
        resampling_freq=resampling_freq,
        clean_ival_ms=clean_ival_ms,
        train_test_split=train_test_split
    )

    # epoching continuous data (from RawArray to SignalAndTarget)
    epo = create_signal_target_from_raw_mne(
        cnt,
        name_to_start_codes,
        epoch_ival_ms
    )
    tot_len = len(epo.y)

    # if train_len is None, train and test data were not split
    if train_len is None:
        train_len = round(tot_len / 1.8)
    test_len = tot_len - train_len

    # determining losses in train and test data
    indexes = arange(tot_len)
    train_indexes = indexes[0:train_len]
    test_indexes = indexes[-test_len:]
    train_clean_trial_mask = clean_trial_mask[train_indexes]
    test_clean_trial_mask = clean_trial_mask[test_indexes]
    new_train_len = train_clean_trial_mask.astype(npint).sum()
    new_test_len = test_clean_trial_mask.astype(npint).sum()

    # cutting epoched signal
    epo.X = epo.X[clean_trial_mask]
    epo.y = epo.y[clean_trial_mask]

    # creating EEGDataset instance and returning it
    return EEGDataset.from_epo_to_dataset(epo=epo,
                                          train_len=new_train_len,
                                          test_len=new_test_len,
                                          validation_frac=validation_frac)
