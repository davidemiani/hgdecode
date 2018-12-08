from numpy import ceil
from numpy import floor
from numpy import zeros
from numpy import arange
from numpy import unique
from numpy import repeat
from numpy import newaxis
from numpy import concatenate
from keras.utils import to_categorical


class FilterBank(object):
    """Create filter bank for FBCSP algorithm.

    Compute frequencies for the filter bank of a specific FBCSP
    experiment.

    Parameters
    ----------
    min_freq : int or list of int
        filter banks min frequency
    max_freq : int or list of int
        filter banks max frequency
    window : int or list of int
        bandwidth of each filter
    overlap : int or list of int

    Returns
    -------
    filter_bank : instance of FilterBank object

    Author info
    -----------
    CREDITS:     Davide Miani (nov 2018)
    LAST REVIEW: Davide Miani (nov 2018)
    MAIL TO:     davide.miani2@gmail.com
    Visit my GitHub to find more:
    https://github.com/davidemiani
    """

    def __init__(self,
                 min_freq=0,
                 max_freq=12,
                 window=6,
                 overlap=3):
        # TODO: validate inputs

        # if multi-input, recalling itself recursively
        if type(min_freq) is list:
            bank = FilterBank(
                min_freq=min_freq[0],
                max_freq=max_freq[0],
                window=window[0],
                overlap=overlap[0]
            )
            for idx in range(1, len(min_freq)):
                bank = FilterBank.merge_banks(
                    bank,
                    FilterBank(min_freq=min_freq[idx],
                               max_freq=max_freq[idx],
                               window=window[idx],
                               overlap=overlap[idx])
                )
            # copying bank as final object
            self.min_freq = min_freq
            self.max_freq = bank.max_freq
            self.window = bank.window
            self.overlap = bank.overlap
            self.bank = bank.bank
        else:
            # copying inputs as properties with same names and values
            self.min_freq = min_freq
            self.max_freq = max_freq
            self.window = window
            self.overlap = overlap

            # computing filter bank
            self.bank = self.compute_bank()

        # fixing 0 frequency (not permitted from scipy.signal.butter)
        if self.bank[0, 0] == 0:
            self.bank[0, 0] = 0.1

    def __repr__(self):
        return repr(self.bank)

    def __str__(self):
        return str(self.bank)

    def __len__(self):
        return len(self.bank)

    def __iter__(self):
        return self.bank.__iter__()

    def __getitem__(self, item):
        return self.bank[item]

    @property
    def shape(self):
        return self.bank.shape

    @staticmethod
    def merge_banks(bank1, bank2):
        bank = bank1
        bank.min_freq = [bank1.min_freq, bank2.min_freq]
        bank.max_freq = [bank1.max_freq, bank2.max_freq]
        bank.window = [bank1.window, bank2.window]
        bank.overlap = [bank1.overlap, bank2.overlap]
        bank.bank = concatenate([bank1.bank, bank2.bank])
        return bank

    def compute_bank(self):
        # computing filter bank length for pre-allocation
        bank_length = int(floor((self.max_freq - self.min_freq) /
                                (self.window - self.overlap)) - 1)

        # pre-allocating numpy array
        bank = zeros((bank_length, 2))

        # determining first init and stop
        init = self.min_freq
        stop = init + self.window

        # cycling on bank to allocate
        for idx in range(bank_length):
            # allocating
            bank[idx, 0] = init
            bank[idx, 1] = stop

            # updating init and stop
            init = stop - self.overlap
            stop = init + self.window

        return bank


class EEGDataset(object):
    """
    # TODO: documentation for this class
    """

    def __init__(self,
                 epo_train_x,
                 epo_train_y,
                 epo_valid_x,
                 epo_valid_y,
                 epo_test_x,
                 epo_test_y):
        assert len(epo_train_x) == len(epo_train_y)
        assert len(epo_valid_x) == len(epo_valid_y)
        assert len(epo_test_x) == len(epo_test_y)
        self.X_train = epo_train_x
        self.y_train = epo_train_y
        self.X_valid = epo_valid_x
        self.y_valid = epo_valid_y
        self.X_test = epo_test_x
        self.y_test = epo_test_y

    def __repr__(self):
        return '<EEGDataset with train:{:d}, valid:{:d}, test:{:d}>'.format(
            len(self.y_train), len(self.y_valid), len(self.y_test)
        )

    def __str__(self):
        return '<EEGDataset with train:{:d}, valid:{:d}, test:{:d}>'.format(
            len(self.y_train), len(self.y_valid), len(self.y_test)
        )

    def __len__(self):
        return len(self.y_train) + len(self.y_valid) + len(self.y_test)

    @property
    def shape(self):
        return self.X_train.shape[1:]

    @property
    def train_frac(self):
        return len(self.y_train) / len(self)

    @property
    def valid_frac(self):
        return len(self.y_valid) / len(self)

    @property
    def test_frac(self):
        return len(self.y_test) / len(self)

    @property
    def n_channels(self):
        return self.shape[0]

    @property
    def n_samples(self):
        return self.shape[1]

    @staticmethod
    def from_epo_to_dataset(epo, train_len, test_len, validation_frac=0.2):
        # computing number of trails for each valid, train & test
        tot_len = len(epo.y)
        valid_len = int(floor(train_len * validation_frac))
        train_len = train_len - valid_len

        # computing indexes
        indexes = arange(tot_len)
        train_indexes = indexes[0:train_len]
        valid_indexes = indexes[train_len:(train_len + valid_len)]
        test_indexes = indexes[-test_len:]

        # cutting epo into train, valid & test
        epo_train_x = epo.X[train_indexes, ...]
        epo_train_y = epo.y[train_indexes, ...]
        epo_valid_x = epo.X[valid_indexes, ...]
        epo_valid_y = epo.y[valid_indexes, ...]
        epo_test_x = epo.X[test_indexes, ...]
        epo_test_y = epo.y[test_indexes, ...]

        return EEGDataset(epo_train_x,
                          epo_train_y,
                          epo_valid_x,
                          epo_valid_y,
                          epo_test_x,
                          epo_test_y)

    def make_crops(self, crop_sample_size=None, crop_stride=None):
        # TODO: validating inputs
        if crop_sample_size is not None:
            # cropping train
            self.X_train, self.y_train = self.crop_x_y(self.X_train,
                                                       self.y_train,
                                                       crop_sample_size,
                                                       crop_stride)
            # cropping valid
            self.X_valid, self.y_valid = self.crop_x_y(self.X_valid,
                                                       self.y_valid,
                                                       crop_sample_size,
                                                       crop_stride)
            # cropping test
            self.X_test, self.y_test = self.crop_x_y(self.X_test,
                                                     self.y_test,
                                                     crop_sample_size,
                                                     crop_stride)

    @staticmethod
    def crop_x_y(x, y, crop_sample_size, crop_stride):
        # getting shapes
        d = x.shape[0]
        h = x.shape[1]
        w = x.shape[2]

        # determining how many crops
        n_crops = int(ceil(
            (w - crop_sample_size + 1) / crop_stride
        ))
        new_d = n_crops * d
        new_h = h
        new_w = crop_sample_size

        # pre-allocating
        new_x = zeros((new_d, new_h, new_w))
        new_y = zeros(new_d)

        # filling pre-allocated arrays
        init = 0
        stop = init + n_crops
        for i in range(d):
            # new_x[init:stop, ...] = EEGDataset.crop_x(x[i, ...],
            #                                        n_crops,
            #                                        crop_sample_size,
            #                                        crop_stride)
            new_y[init:stop, ...] = EEGDataset.crop_y(y[i], n_crops)

            # updating init & stop
            init = init + n_crops
            stop = stop + n_crops

        # returning new arrays
        return new_x, new_y

    @staticmethod
    def crop_x(x, n_crops, crop_sample_size, crop_stride):
        return x

    @staticmethod
    def crop_y(y, n_crops):
        return repeat(y, n_crops)

    def add_axis(self):
        self.X_train = self.X_train[..., newaxis]
        self.X_valid = self.X_valid[..., newaxis]
        self.X_test = self.X_test[..., newaxis]

    def to_categorical(self, n_classes=None):
        if n_classes is None:
            n_classes = len(unique(self.y_train))
        self.y_train = to_categorical(self.y_train, n_classes)
        self.y_valid = to_categorical(self.y_valid, n_classes)
        self.y_test = to_categorical(self.y_test, n_classes)
