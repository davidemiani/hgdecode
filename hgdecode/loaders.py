import logging as log
from copy import deepcopy
from numpy import max
from numpy import abs
from numpy import sum
from numpy import std
from numpy import mean
from numpy import array
from numpy import floor
from numpy import repeat
from numpy import arange
from numpy import setdiff1d
from numpy import concatenate
from numpy import count_nonzero
from numpy.random import RandomState
from os.path import join
from mne.io.array.array import RawArray
from hgdecode.utils import print_manager
from hgdecode.classes import CrossValidation
from sklearn.model_selection import StratifiedKFold
from braindecode.datasets.bbci import BBCIDataset
from braindecode.mne_ext.signalproc import mne_apply
from braindecode.mne_ext.signalproc import resample_cnt
from braindecode.mne_ext.signalproc import concatenate_raws_with_events
from braindecode.datautil.signalproc import bandpass_cnt
from braindecode.datautil.signalproc import exponential_running_standardize
from braindecode.datautil.signal_target import SignalAndTarget
from braindecode.datautil.trial_segment import \
    create_signal_target_from_raw_mne


# TODO: re-implement all this functions as an unique class


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


def pick_right_channels(cnt, channel_names):
    # return the same cnt but with only right channels
    return cnt.pick_channels(channel_names)


def standardize_cnt(cnt, standardize_mode=0):
    # computing frequencies
    sampling_freq = cnt.info['sfreq']
    init_freq = 0.1
    stop_freq = sampling_freq / 2 - 0.1
    filt_order = 3
    axis = 0
    filtfilt = False

    # filtering DC and frequencies higher than the nyquist one
    cnt = mne_apply(
        lambda x:
        bandpass_cnt(
            data=x,
            low_cut_hz=init_freq,
            high_cut_hz=stop_freq,
            fs=sampling_freq,
            filt_order=filt_order,
            axis=axis,
            filtfilt=filtfilt
        ),
        cnt
    )

    # removing mean and normalizing in 3 different ways
    if standardize_mode == 0:
        # x - mean
        cnt = mne_apply(
            lambda x:
            x - mean(x, axis=0, keepdims=True),
            cnt
        )
    elif standardize_mode == 1:
        # (x - mean) / std
        cnt = mne_apply(
            lambda x:
            (x - mean(x, axis=0, keepdims=True)) /
            std(x, axis=0, keepdims=True),
            cnt
        )
    elif standardize_mode == 2:
        # parsing to milli volt for numerical stability of next operations
        cnt = mne_apply(lambda a: a * 1e6, cnt)

        # applying exponential_running_standardize (Schirrmeister)
        cnt = mne_apply(
            lambda x:
            exponential_running_standardize(
                x.T,
                factor_new=1e-3,
                init_block_size=1000,
                eps=1e-4
            ).T,
            cnt
        )
    return cnt


def load_and_preprocess_data(data_dir,
                             name_to_start_codes,
                             channel_names,
                             subject_id=1,
                             resampling_freq=None,
                             clean_ival_ms=(0, 4000),
                             train_test_split=True,
                             clean_on_all_channels=True,
                             standardize_mode=0):
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
    print_manager('DATA LOADING ROUTINE FOR SUBJ ' + str(subject_id),
                  'double-dashed')
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
    log.info('Standardize mode: {}'.format(standardize_mode))
    cnt = standardize_cnt(cnt=cnt, standardize_mode=standardize_mode)
    print_manager('DONE!!', 'last', bottom_return=1)

    return cnt, clean_trial_mask


def ml_loader(data_dir,
              name_to_start_codes,
              channel_names,
              subject_id=1,
              resampling_freq=None,
              clean_ival_ms=(0, 4000),
              train_test_split=True,
              clean_on_all_channels=True,
              standardize_mode=0):
    outputs = load_and_preprocess_data(
        data_dir=data_dir,
        name_to_start_codes=name_to_start_codes,
        channel_names=channel_names,
        subject_id=subject_id,
        resampling_freq=resampling_freq,
        clean_ival_ms=clean_ival_ms,
        train_test_split=train_test_split,
        clean_on_all_channels=clean_on_all_channels,
        standardize_mode=standardize_mode
    )
    return outputs[0], outputs[1]


def dl_loader(data_dir,
              name_to_start_codes,
              channel_names,
              subject_id=1,
              resampling_freq=None,
              clean_ival_ms=(0, 4000),
              epoch_ival_ms=(-500, 4000),
              train_test_split=True,
              clean_on_all_channels=True,
              standardize_mode=0):
    # loading and pre-processing data
    cnt, clean_trial_mask = load_and_preprocess_data(
        data_dir=data_dir,
        name_to_start_codes=name_to_start_codes,
        channel_names=channel_names,
        subject_id=subject_id,
        resampling_freq=resampling_freq,
        clean_ival_ms=clean_ival_ms,
        train_test_split=train_test_split,
        clean_on_all_channels=clean_on_all_channels,
        standardize_mode=standardize_mode
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


class CrossSubject(object):
    """
    Nel momento in cui si crea una istanza di questa classe, essa caricherà
    tutti quanti gli id soggetto indicati in formato cnt con le relative
    maschere ed un nuovo array che ci dice tutti gli indici in cui iniziano
    i vari soggetti; inoltre, si andrà poi a specificare volta per volta il
    tipo di dato che si vorrà avere in memoria utilizzando un metodo parser,
    che andrà a sovrascrivere i dati nel nuovo formato.
    """

    def __init__(self,
                 data_dir,
                 subject_ids,
                 channel_names,
                 name_to_start_codes,
                 random_state=None,
                 validation_frac=None,
                 validation_size=None,
                 resampling_freq=None,
                 train_test_split=True,
                 clean_ival_ms=(-500, 4000),
                 epoch_ival_ms=(-500, 4000),
                 clean_on_all_channels=True):
        # from input properties
        self.data_dir = data_dir
        self.subject_ids = subject_ids
        self.channel_names = channel_names
        self.name_to_start_codes = name_to_start_codes
        self.resampling_freq = resampling_freq
        self.train_test_split = train_test_split
        self.clean_ival_ms = clean_ival_ms
        self.epoch_ival_ms = epoch_ival_ms
        self.clean_on_all_channels = clean_on_all_channels

        # saving random state; if it is not specified, creating a 1234 one
        if random_state is None:
            self.random_state = RandomState(1234)
        else:
            self.random_state = random_state

        # other object properties
        self.data = None
        self.clean_trial_mask = []
        self.subject_labels = None
        self.subject_indexes = []
        self.folds = None

        # for fold specific data, creating a blank property
        self.fold_data = None
        self.fold_subject_labels = None

        # loading the first subject (to pre-allocate cnt array)
        temp_cnt, temp_mask = load_and_preprocess_data(
            data_dir=self.data_dir,
            name_to_start_codes=self.name_to_start_codes,
            channel_names=self.channel_names,
            subject_id=self.subject_ids[0],
            resampling_freq=self.resampling_freq,
            clean_ival_ms=self.clean_ival_ms,
            train_test_split=self.train_test_split,
            clean_on_all_channels=self.clean_on_all_channels
        )

        # allocate the first subject_labels
        temp_labels = repeat(array([subject_ids[0]]), len(temp_mask))

        # appending new indexes (only cleaned cnt will count!)
        last_non_zero_len = count_nonzero(self.clean_trial_mask)
        self.subject_indexes.append(
            [last_non_zero_len, last_non_zero_len + count_nonzero(temp_mask)]
        )

        # merging cnt, mask and labels (in this case assigning)
        self.data = temp_cnt
        self.clean_trial_mask = temp_mask
        self.subject_labels = temp_labels

        # creating iterable object from subject_ids and skipping the first one
        iter_subjects = iter(self.subject_ids)
        next(iter_subjects)

        # loading all others cnt data and concatenating them
        for current_subject in iter_subjects:
            # loading current subject cnt and mask
            temp_cnt, temp_mask = load_and_preprocess_data(
                subject_id=current_subject,  # here the current subject!
                data_dir=self.data_dir,
                name_to_start_codes=self.name_to_start_codes,
                channel_names=self.channel_names,
                resampling_freq=self.resampling_freq,
                clean_ival_ms=self.clean_ival_ms,
                train_test_split=self.train_test_split,
                clean_on_all_channels=self.clean_on_all_channels
            )

            # create the subject_labels for this subject
            temp_labels = repeat(array([current_subject]), len(temp_mask))

            # appending new indexes (only cleaned cnt will count!)
            last_non_zero_len = count_nonzero(self.clean_trial_mask)
            self.subject_indexes.append(
                [last_non_zero_len,
                 last_non_zero_len + count_nonzero(temp_mask)]
            )

            # merging cnt and mask
            self.data = concatenate_raws_with_events([self.data, temp_cnt])
            self.clean_trial_mask = \
                concatenate([self.clean_trial_mask, temp_mask])
            self.subject_labels = \
                concatenate([self.subject_labels, temp_labels])

        # computing validation_frac and validation_size
        if validation_size is None:
            if validation_frac is None:
                self.validation_frac = 0
                self.validation_size = 0
            else:
                self.validation_frac = validation_frac
                self.validation_size = \
                    int(floor(self.n_trials * self.validation_frac))
        else:
            self.validation_size = validation_size
            self.validation_frac = self.validation_size / self.n_trials

    def parser(self, output_format, leave_subj=None, parsing_type=0):
        """
        HOW DOES IT WORK?
        -----------------
        if parsing_type is 0 then epoched signal will be saved in
        fold_data; if parsing_type is 1 then the cnt signal will be replaced
        with the epoched one
        """
        if output_format is 'epo':
            self.cnt_to_epo(parsing_type=parsing_type)
        elif output_format is 'EEGDataset':
            self.cnt_to_epo(parsing_type=parsing_type)
            self.epo_to_dataset(leave_subj=leave_subj,
                                parsing_type=parsing_type)

    def cnt_to_epo(self, parsing_type):
        # checking if data is cnt; if not, the method will not work
        if isinstance(self.data, RawArray):
            """
            WHATS GOING ON HERE?
            --------------------
            If parsing_type is 0, then there will be a 'soft parsing
            routine', data will parsed and stored in fold_data instead of
            in the main data property
            """
            if parsing_type == 0:
                # parsing from cnt to epoch
                print_manager('Parsing cnt signal to epoched one...')
                self.fold_data = create_signal_target_from_raw_mne(
                    self.data,
                    self.name_to_start_codes,
                    self.epoch_ival_ms
                )
                print_manager('DONE!!', bottom_return=1)

                # cleaning signal and labels with mask
                print_manager('Cleaning epoched signal with mask...')
                self.fold_data.X = self.fold_data.X[self.clean_trial_mask]
                self.fold_data.y = self.fold_data.y[self.clean_trial_mask]
                self.fold_subject_labels = \
                    self.subject_labels[self.clean_trial_mask]
                print_manager('DONE!!', bottom_return=1)
            elif parsing_type == 1:
                """
                WHATS GOING ON HERE?
                --------------------
                If parsing_type is 1, then the epoched signal will replace 
                the original one in the data property
                """
                print_manager('Parsing cnt signal to epoched one...')
                self.data = create_signal_target_from_raw_mne(
                    self.data,
                    self.name_to_start_codes,
                    self.epoch_ival_ms
                )
                print_manager('DONE!!', bottom_return=1)

                # cleaning signal and labels
                print_manager('Cleaning epoched signal with mask...')
                self.data.X = self.data.X[self.clean_trial_mask]
                self.data.y = self.data.y[self.clean_trial_mask]
                self.subject_labels = \
                    self.subject_labels[self.clean_trial_mask]
                print_manager('DONE!!', bottom_return=1)
            else:
                raise ValueError(
                    'parsing_type {} not supported.'.format(parsing_type)
                )

            # now that we have an epoched signal, we can already create
            # folds for cross-subject validation
            self.create_balanced_folds()

    def epo_to_dataset(self, leave_subj, parsing_type=0):
        print_manager('FOLD ALL BUT ' + str(leave_subj), 'double-dashed')
        print_manager('Creating current fold...')

        print_manager('DONE!!', bottom_return=1)
        print_manager('Parsing epoched signal to EEGDataset...')
        if parsing_type is 0:
            self.fold_data = CrossValidation.create_dataset_static(
                self.fold_data, self.folds[leave_subj - 1]
            )
        elif parsing_type is 1:
            self.fold_data = CrossValidation.create_dataset_static(
                self.data, self.folds[leave_subj - 1]
            )
        else:
            raise ValueError(
                'parsing_type {} not supported.'.format(parsing_type)
            )
        print_manager('DONE!!', bottom_return=1)
        print_manager('We obtained a ' + str(self.fold_data))
        print_manager('DATA READY!!', 'last', bottom_return=1)

    def create_balanced_folds(self):
        # pre-allocating folds
        self.folds = []
        for subj_idx, subj_idxs in enumerate(self.subject_indexes):
            # getting current test_idxs (all a subject trials)
            test_idxs = arange(subj_idxs[0], subj_idxs[1])

            # getting train_idxs as all but the current subject
            train_idxs = setdiff1d(arange(self.n_trials), test_idxs)

            # pre-allocating valid_idxs
            valid_idxs = array([], dtype='int')

            # if no validation set is required...
            if self.validation_frac == 0:
                # setting valid_idxs to None, else...
                valid_idxs = None
            else:
                # ...determining number of splits for this train/validation set
                n_splits = int(floor(self.validation_frac * 100))

                # getting StratifiesKFold object
                skf = StratifiedKFold(n_splits=n_splits,
                                      random_state=self.random_state,
                                      shuffle=True)

                # cycling on subject in the train fold
                for c_subj_idx, c_subj_idxs in enumerate(self.subject_indexes):
                    if c_subj_idx == subj_idx:
                        # nothing to do
                        pass
                    else:
                        # splitting first subject train / valid
                        X, y = self._get_subject_data(c_subj_idx)

                        # get batch from StratifiedKFold object
                        for c_train_idxs, c_valid_idxs in skf.split(X=X, y=y):
                            # referring c_train_idxs and c_valid_idxs
                            c_train_idxs += c_subj_idxs[0]
                            c_valid_idxs += c_subj_idxs[0]

                            # remove this batch indexes from train_idxs
                            train_idxs = setdiff1d(train_idxs, c_valid_idxs)

                            # adding this batch indexes to valid_idxs
                            valid_idxs = concatenate([valid_idxs,
                                                      c_valid_idxs])
                            # all is done for this subject!! Breaking cycle
                            break

            # appending new fold
            self.folds.append(
                {
                    'train': train_idxs,
                    'valid': valid_idxs,
                    'test': test_idxs
                }
            )

    def _get_subject_data(self, subj_idx):
        init = self.subject_indexes[subj_idx][0]
        stop = self.subject_indexes[subj_idx][1]
        ival = arange(init, stop)
        if isinstance(self.fold_data, SignalAndTarget):
            return self.fold_data.X[ival], self.fold_data.y[ival]
        elif isinstance(self.data, SignalAndTarget):
            return self.data.X[ival], self.data.y[ival]
        else:
            raise ValueError('You are trying to get epoched data but you '
                             'still have to parse cnt data.')

    @property
    def n_trials(self):
        return count_nonzero(self.clean_trial_mask)
