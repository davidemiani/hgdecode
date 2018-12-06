from numpy import max
from numpy import abs
from numpy import sum
from numpy import mean
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


def load_cnt(file_path):
    # create the loader object for BBCI standard
    loader = BBCIDataset(file_path, load_sensor_names=None)

    # load data
    return loader.load()


def get_clean_trial_mask(cnt, name_to_start_codes, clean_interval=(0, 4000)):
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
        clean_interval
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


def picking_right_channels(cnt, channel_names):
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


def ml_loader(data_dir,
              name_to_start_codes,
              channel_names,
              subject_id=1,
              resampling_freq=250,
              clean_interval=(0, 4000),
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
    elif len(file_paths) == 2:
        # loading train_cnt, test_cnt and merging them
        train_cnt = load_cnt(file_paths[0])
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
        clean_interval
    )
    print_manager('DONE!!', bottom_return=1)

    # pick only right channels
    log.info('Picking only right channels...')
    cnt = picking_right_channels(cnt, channel_names)
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

    #
    print_manager(
        'LOADING ROUTINE COMPLETED',
        'last',
        bottom_return=1
    )

    # return train_data (now complete) as dataset
    return cnt, clean_trial_mask


def dl_loader(data_dir,
              name_to_start_codes,
              channel_names,
              subject_id=1,
              resampling_freq=250,
              clean_interval=(0, 4000),
              train_test_split=True):
    # getting data paths
    file_paths = get_data_files_paths(
        data_dir,
        subject_id=subject_id,
        train_test_split=train_test_split
    )

# TODO: dl_loader
