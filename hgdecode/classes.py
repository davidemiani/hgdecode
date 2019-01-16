import sys
import time
import matplotlib.pyplot as plt
from pylab import savefig
from numpy import std
from numpy import ceil
from numpy import mean
from numpy import log10
from numpy import floor
from numpy import zeros
from numpy import round
from numpy import arange
from numpy import unique
from numpy import repeat
from numpy import append
from numpy import random
from numpy import newaxis
from numpy import linspace
from numpy import setdiff1d
from numpy import concatenate
from numpy.random import RandomState
from pickle import dump
from pickle import load
from os.path import join
from os.path import exists
from os.path import basename
from collections import OrderedDict
from keras.utils import Sequence
from keras.utils import to_categorical
from keras.callbacks import Callback
from hgdecode.utils import csv_manager
from hgdecode.utils import print_manager
from sklearn.metrics import confusion_matrix
from braindecode.datautil.iterators import get_balanced_batches


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
        # TODO: is it deprecated? Consider to remove this method.
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

    def make_crops(self, crop_sample_size=None, crop_step=None):
        # TODO: validating inputs
        if crop_sample_size is not None:
            # printing
            print_manager('CROPPING ROUTINE', 'double-dashed')

            # cropping train
            print_manager('Cropping train...')
            self.X_train, self.y_train = self.crop_X_y(self.X_train,
                                                       self.y_train,
                                                       crop_sample_size,
                                                       crop_step)
            print_manager('DONE!!', bottom_return=1)

            # cropping valid
            print_manager('Cropping validation...')
            self.X_valid, self.y_valid = self.crop_X_y(self.X_valid,
                                                       self.y_valid,
                                                       crop_sample_size,
                                                       crop_step)
            print_manager('DONE!!', bottom_return=1)

            # cropping test
            print_manager('Cropping test...')
            self.X_test, self.y_test = self.crop_X_y(self.X_test,
                                                     self.y_test,
                                                     crop_sample_size,
                                                     crop_step)
            print_manager('DONE!!', 'last', bottom_return=1)

    @staticmethod
    def crop_X_y(X, y, crop_sample_size, crop_step):
        # getting shapes
        d = X.shape[0]
        h = X.shape[1]
        w = X.shape[2]

        # determining how many crops
        n_crops = int(ceil(
            (w - crop_sample_size + 1) / crop_step
        ))
        new_d = n_crops * d
        new_h = h
        new_w = crop_sample_size

        # pre-allocating
        new_X = zeros((new_d, new_h, new_w))
        new_y = zeros(new_d)

        # filling pre-allocated arrays
        init = 0
        stop = init + n_crops
        for i in range(d):
            new_X[init:stop, ...] = EEGDataset.crop_X(X[i, ...],
                                                      n_crops,
                                                      crop_sample_size,
                                                      crop_step)
            new_y[init:stop, ...] = EEGDataset.crop_y(y[i], n_crops)

            # updating init & stop
            init = init + n_crops
            stop = stop + n_crops

        # returning new arrays
        return new_X, new_y

    @staticmethod
    def crop_X(X, n_crops, crop_sample_size, crop_step):
        # pre-allocating new_x
        new_X = zeros((n_crops, X.shape[0], crop_sample_size))

        # handling init & stop in X array
        init = 0
        stop = crop_sample_size

        # cycling on new_x depth
        for i in range(n_crops):
            new_X[i, ...] = X[:, init:stop]
            init = init + crop_step
            stop = stop + crop_step

        # returning new_x
        return new_X

    @staticmethod
    def crop_y(y, n_crops):
        return repeat(y, n_crops)

    def add_axis(self):
        # TODO: channel first or last
        self.X_train = self.X_train[:, newaxis, ...]
        self.X_valid = self.X_valid[:, newaxis, ...]
        self.X_test = self.X_test[:, newaxis, ...]

    def to_categorical(self, n_classes=None):
        if n_classes is None:
            n_classes = len(unique(self.y_train))
        self.y_train = to_categorical(self.y_train, n_classes)
        self.y_valid = to_categorical(self.y_valid, n_classes)
        self.y_test = to_categorical(self.y_test, n_classes)


class EEGDataGenerator(Sequence):
    """
    # TODO: class description
    """

    def __init__(self,
                 # data
                 X,
                 y,
                 # main dimensions
                 batch_size=512,
                 n_classes=None,
                 # crop dimensions
                 crop_sample_size=512,
                 crop_step=1,
                 # others
                 shuffle=True):
        """Initialization"""
        # data
        self.X = X
        self.y = y

        # main dimensions
        self.batch_size = batch_size
        self.n_trials = X.shape[0]
        self.n_channels = X.shape[1]
        self.n_samples = X.shape[2]
        if n_classes is None:
            self.n_classes = len(unique(y))
        else:
            self.n_classes = n_classes

        # crop dimensions
        self.crop_sample_size = crop_sample_size
        self.crop_step = crop_step
        self.n_crops_for_trial = int(ceil(
            (self.n_samples - crop_sample_size + 1) / crop_step
        ))
        self.n_crops = self.n_crops_for_trial * self.n_trials

        # others
        self.shuffle = shuffle
        self.current_batch = None

        # allocating indexes to None, than updating using on_epoch_end()
        self.indexes = None  # pointer to trials
        self.next_to_unpack = None  # pointer to indexes
        self.on_epoch_end()

        # pre-allocating crop stack & unpacking the first trial indexed
        self.crop_stack_X = None
        self.crop_stack_y = None
        self.unpack_trial()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(floor(self.n_crops / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # registering current batch
        self.current_batch = index

        # Generate data
        X, y = self.__data_generation()

        return X, y

    def on_epoch_end(self):
        """
        Updates indexes after each epoch; the indexes order will tell the
        trainer what is the next trial order to unpack
        """
        # TODO: set seed rng
        self.indexes = arange(self.n_trials)
        if self.shuffle is True:
            random.shuffle(self.indexes)
        self.next_to_unpack = 0

    def __data_generation(self):
        """Generates data containing batch_size samples"""
        while len(self.crop_stack_y) < self.batch_size:
            self.unpack_trial()

        # getting first batch_size elements from stacks
        start = 0
        stop = self.batch_size
        indexes = arange(start=start, stop=stop)
        X = self.crop_stack_X[indexes, ...]
        y = self.crop_stack_y[indexes]

        # popping stack
        start = self.batch_size
        stop = len(self.crop_stack_y)
        indexes = arange(start=start, stop=stop)
        self.crop_stack_X = self.crop_stack_X[indexes, ...]
        self.crop_stack_y = self.crop_stack_y[indexes]

        # forcing the x examples to have 4 dimensions
        X = X[:, newaxis, ...]

        # parsing y to categorical
        y = to_categorical(y, num_classes=self.n_classes)

        # returning data generated
        return X, y

    def unpack_trial(self):
        # first unpack has to (re)create the stack
        if self.next_to_unpack is 0:
            # unpacking first X trial
            self.crop_stack_X = EEGDataset.crop_X(
                self.X[self.indexes[self.next_to_unpack], ...],
                self.n_crops_for_trial, self.crop_sample_size, self.crop_step
            )

            # unpacking first y trial
            self.crop_stack_y = EEGDataset.crop_y(
                self.y[self.indexes[self.next_to_unpack]],
                self.n_crops_for_trial
            )
        else:
            # appending X
            self.crop_stack_X = append(
                self.crop_stack_X,
                EEGDataset.crop_X(
                    self.X[self.indexes[self.next_to_unpack], ...],
                    self.n_crops_for_trial,
                    self.crop_sample_size,
                    self.crop_step
                ),
                axis=0
            )

            # appending y
            self.crop_stack_y = append(
                self.crop_stack_y,
                EEGDataset.crop_y(
                    self.y[self.indexes[self.next_to_unpack]],
                    self.n_crops_for_trial
                )
            )

        # updating next_to_unpack
        self.next_to_unpack += 1


class MetricsTracker(Callback):
    def __init__(self,
                 dataset,
                 epochs,
                 n_classes,
                 batch_size,
                 h5_model_path,
                 fold_stats_path):
        # allocating inputs as properties
        self.dataset = dataset
        self.epochs = epochs
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.h5_model_path = h5_model_path
        self.fold_stats_path = fold_stats_path

        # pre-allocating train, valid and test dicts with loss and conf_mtx
        self.train = {'loss': [],
                      'acc': []}
        self.valid = {'loss': [],
                      'acc': []}

        # pre-allocating best (to track the best net configuration)
        self.best = {'loss': float('inf'),
                     'acc': 0,
                     'idx': None}

        # calling the super class constructor
        Callback.__init__(self)

    def on_epoch_end(self, epoch, logs={}):
        print('Computing statistics on this epoch:')
        epoch_string_length = len(str(self.epochs)) * 2
        progress_bar_length = 30 + epoch_string_length
        progress_bar = ProgressBar(target=2, width=progress_bar_length)

        # computing loss and accuracy on train
        progress_bar.update(current=0, message='evaluating train')
        score = self.model.evaluate(
            x=self.dataset.X_train,
            y=self.dataset.y_train,
            batch_size=self.batch_size,
            verbose=0
        )
        self.train['loss'].append(score[0])
        self.train['acc'].append(score[1])

        # getting loss and accuracy on validation from logs
        progress_bar.update(current=1, message='evaluating valid')
        self.valid['loss'].append(logs.get('val_loss'))
        self.valid['acc'].append(logs.get('val_acc'))

        # updating prog bar for the end
        message = 'loss: {0:.4f}'.format(self.train['loss'][epoch]) + \
                  ' - acc: {0:.4f}'.format(self.train['acc'][epoch]) + \
                  ' - val_loss: {0:.4f}'.format(self.valid['loss'][epoch]) + \
                  ' - val_acc: {0:.4f}'.format(self.valid['acc'][epoch])
        progress_bar.update(current=2, message=message)

        # if this is the best net, saving it
        if self.valid['loss'][epoch] <= self.best['loss']:
            if self.valid['acc'][epoch] > self.best['acc']:
                print('New best model found!! :-D\n')
                self.model.save(self.h5_model_path)
                self.best['idx'] = epoch
                self.best['loss'] = self.valid['loss'][epoch]
                self.best['acc'] = self.valid['acc'][epoch]

    def on_train_end(self, logs=None):
        # printing the end of training
        print_manager('TRAINING ENDED', 'last', bottom_return=1)

        # printing the start of testing
        print_manager('RUNNING TESTING', 'double-dashed')

        # loading best net
        self.model.load_weights(self.h5_model_path)

        # running test
        test_loss, test_acc = self.model.evaluate(
            self.dataset.X_test,
            self.dataset.y_test,
            verbose=1
        )
        print('Test loss:', test_loss)
        print('Test  acc:', test_acc)

        # making predictions on X_test with final model and getting also
        # y_test from memory; parsing both back from categorical
        y_test = self.dataset.y_test.argmax(axis=1)
        y_pred = self.model.predict(self.dataset.X_test).argmax(axis=1)

        # computing confusion matrix
        conf_mtx = confusion_matrix(y_true=y_test, y_pred=y_pred)
        print("\nConfusion matrix:\n", conf_mtx)

        # creating results dictionary
        results = {
            'train': {
                'loss': self.train['loss'],
                'acc': self.train['acc']
            },
            'valid': {
                'loss': self.valid['loss'],
                'acc': self.valid['acc']
            },
            'best': {
                'loss': self.best['loss'],
                'acc': self.best['acc'],
                'idx': self.best['idx']
            },
            'test': {
                'loss': test_loss,
                'acc': test_acc,
                'conf_mtx': conf_mtx.tolist()
            }
        }

        # dumping and saving
        with open(self.fold_stats_path, 'wb') as f:
            dump(results, f)

        # printing the end
        print_manager('TESTING ENDED', 'last', bottom_return=1)


class ProgressBar(object):
    """Displays a progress bar.
    # Arguments
        target: Total number of steps expected, None if unknown.
        width: Progress bar width on screen.
        verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
        stateful_metrics: Iterable of string names of metrics that
            should *not* be averaged over time. Metrics in this list
            will be displayed as-is. All others will be averaged
            by the progress bar before display.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=30, interval=0.05):
        self.target = target
        self.width = width
        self.interval = interval
        self._dynamic_display = ((hasattr(sys.stdout, 'isatty') and
                                  sys.stdout.isatty()) or
                                 'ipykernel' in sys.modules)
        self._total_width = 0
        self._seen_so_far = 0
        self._values = OrderedDict()
        self._start = time.time()
        self._last_update = 0

    def update(self, current, message=''):
        """Updates the progress bar.
        # Arguments
            current: Index of current step.
            message: Message to display at the end of the bar.
        """
        now = time.time()
        info = ' - %.0fs' % (now - self._start)

        if (now - self._last_update < self.interval and
                self.target is not None and current < self.target):
            return

        prev_total_width = self._total_width
        if self._dynamic_display:
            sys.stdout.write('\b' * prev_total_width)
            sys.stdout.write('\r')
        else:
            sys.stdout.write('\n')

        if self.target is not None:
            numdigits = int(floor(log10(self.target))) + 1
            barstr = '%%%dd/%d [' % (numdigits, self.target)
            bar = barstr % current
            prog = float(current) / self.target
            prog_width = int(self.width * prog)
            if prog_width > 0:
                bar += ('=' * (prog_width - 1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.' * (self.width - prog_width))
            bar += ']'
        else:
            bar = '%7d/Unknown' % current

        self._total_width = len(bar)
        sys.stdout.write(bar)

        if current:
            time_per_unit = (now - self._start) / current
        else:
            time_per_unit = 0
        if self.target is not None and current < self.target:
            eta = time_per_unit * (self.target - current)
            if eta > 3600:
                eta_format = ('%d:%02d:%02d' %
                              (eta // 3600, (eta % 3600) // 60, eta % 60))
            elif eta > 60:
                eta_format = '%d:%02d' % (eta // 60, eta % 60)
            else:
                eta_format = '%ds' % eta

            info = ' - ETA: %s' % eta_format
        else:
            if time_per_unit >= 1:
                info += ' %.0fs/step' % time_per_unit
            elif time_per_unit >= 1e-3:
                info += ' %.0fms/step' % (time_per_unit * 1e3)
            else:
                info += ' %.0fus/step' % (time_per_unit * 1e6)

        if message is not '':
            info += ' - ' + message

        self._total_width += len(info)
        if prev_total_width > self._total_width:
            info += (' ' * (prev_total_width - self._total_width))

        if self.target is not None and current >= self.target:
            info += '\n'

        sys.stdout.write(info)
        sys.stdout.flush()

        self._last_update = now


class CrossValidation(object):
    def __init__(self,
                 epo,
                 n_folds=8,
                 validation_frac=0.1,
                 random_seed=None,
                 shuffle=True):
        # generating random seed if not an input
        if random_seed is None:
            random_seed = RandomState(1234)

        # getting trial number
        n_trials = len(epo.y)

        # getting pseudo-random folds
        folds = get_balanced_batches(
            n_trials=n_trials,
            rng=random_seed,
            shuffle=shuffle,
            n_batches=n_folds
        )

        # train is everything except fold; test is fold indexes
        self.folds = [
            {
                'train': setdiff1d(arange(n_trials), fold),
                'valid': None,
                'test': fold
            }
            for fold in folds
        ]

        # getting validation and reshaping train
        for idx, current_fold in enumerate(self.folds):
            validation_size = int(round(n_trials * validation_frac))
            self.folds[idx]['valid'] = current_fold['train'][-validation_size:]
            self.folds[idx]['train'] = current_fold['train'][:-validation_size]

    @staticmethod
    def create_dataset_for_fold(epo, fold):
        return EEGDataset(
            epo_train_x=epo.X[fold['train'], ...],
            epo_train_y=epo.y[fold['train']],
            epo_valid_x=epo.X[fold['valid'], ...],
            epo_valid_y=epo.y[fold['valid']],
            epo_test_x=epo.X[fold['test'], ...],
            epo_test_y=epo.y[fold['test']]
        )

    @staticmethod
    def cross_validate(subj_results_dir, figures_dir, tables_dir):
        # printing the start
        print_manager('RUNNING CROSS-VALIDATION', 'double-dashed')

        # getting fold file paths
        file_paths = CrossValidation.fold_file_paths(subj_results_dir)

        # figures
        CrossValidation.figures_manager(file_paths, figures_dir)

        # tables
        CrossValidation.tables_manager(file_paths, tables_dir)

        # printing the end
        print_manager('CROSS-VALIDATION ENDED', 'last', bottom_return=1)

    @staticmethod
    def fold_file_paths(subj_results_dir):
        from os import listdir
        from os.path import join
        # getting all fold directories paths
        fold_list = listdir(subj_results_dir)
        fold_list.sort()

        # deleting unuseful stuff
        if '.DS_Store' in fold_list:
            fold_list.remove('.DS_Store')

        # getting all pickle results paths
        file_paths = [join(subj_results_dir, fold_name, 'fold_stats.pickle')
                      for fold_name in fold_list]

        # returning fold file paths
        return file_paths

    @staticmethod
    def load_results(file_paths, loss_acc, train_valid):
        # opening the first file to get n_epochs
        with open(file_paths[0], 'rb') as f:
            results = load(f)
        n_epochs = len(results[train_valid][loss_acc])

        # getting also n_folds
        n_folds = len(file_paths)

        # pre-allocating numpy array
        results = zeros([n_folds, n_epochs])

        # filling all results array
        for idx, file in enumerate(file_paths):
            # opening current results file
            with open(file, 'rb') as f:
                all_results = load(f)

            # appending loss and accuracy for this file
            results[idx, :] = all_results[train_valid][loss_acc]

        # returning results
        return results

    @staticmethod
    def figures_manager(file_paths, figures_dir):
        # plotting train loss
        CrossValidation.internal_plot(
            file_paths=file_paths,
            figures_dir=figures_dir,
            loss_acc='loss',
            train_valid='train'
        )

        # plotting train accuracy
        CrossValidation.internal_plot(
            file_paths=file_paths,
            figures_dir=figures_dir,
            loss_acc='acc',
            train_valid='train'
        )

        # plotting validation loss
        CrossValidation.internal_plot(
            file_paths=file_paths,
            figures_dir=figures_dir,
            loss_acc='loss',
            train_valid='valid'
        )

        # plotting validation accuracy
        CrossValidation.internal_plot(
            file_paths=file_paths,
            figures_dir=figures_dir,
            loss_acc='acc',
            train_valid='valid'
        )

        # plotting loss (both training and validation)
        CrossValidation.comparison_internal_plot(
            file_paths=file_paths,
            figures_dir=figures_dir,
            loss_acc='loss',
            use_std=False
        )

        # plotting acc (both training and validation)
        CrossValidation.comparison_internal_plot(
            file_paths=file_paths,
            figures_dir=figures_dir,
            loss_acc='acc',
            use_std=False
        )

    @staticmethod
    def internal_plot(file_paths,
                      figures_dir,
                      loss_acc='loss',
                      train_valid='train'):
        # getting subject string and creating plot path
        subj_str = basename(figures_dir)
        plot_path = join(figures_dir, subj_str + '_' +
                         train_valid + '_' + loss_acc)

        # loading y values
        y = CrossValidation.load_results(file_paths, loss_acc, train_valid)
        y_mean = mean(y, axis=0)
        y_std = std(y, axis=0)
        n = len(y_mean)

        # creating x values
        x = linspace(1, n, n)

        # setting loss_acc plot properties
        if loss_acc == 'loss':
            y_max = ceil(max(y_mean + y_std))
            y_label = 'loss'
        elif loss_acc == 'acc':
            y_max = 1
            y_label = 'accuracy'
        else:
            raise Exception(
                'loss_acc value "{}" not supported'.format(loss_acc)
            )

        # setting train_valid plot properties
        if train_valid == 'train':
            color = 'red'
            title = 'training set'
        elif train_valid == 'valid':
            color = 'blue'
            title = 'validation set'
        else:
            raise Exception(
                'train_valid value "{}" not supported'.format(train_valid)
            )

        # plotting
        plt.figure(dpi=100, figsize=(12.8, 7.2), facecolor='w', edgecolor='k')
        plt.style.use('seaborn-whitegrid')
        plt.plot(x, y_mean, '-', color=color)
        plt.fill_between(x, y_mean - y_std, y_mean + y_std,
                         color=color, alpha=0.3333)
        plt.xlim(1, n)
        plt.ylim(0, y_max)
        plt.legend(labels=['mean', r'mean$\pm$std'], fontsize=20, frameon=True)
        plt.xlabel('epochs', fontsize=25)
        plt.ylabel(y_label, fontsize=25)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.title(title, fontsize=25)

        # saving figure
        savefig(plot_path)

    @staticmethod
    def comparison_internal_plot(file_paths,
                                 figures_dir,
                                 loss_acc='loss',
                                 use_std=False):
        # getting subject string and creating plot path
        subj_str = basename(figures_dir)
        plot_path = join(figures_dir, subj_str + '_' + loss_acc)

        # loading y values
        y_train = CrossValidation.load_results(file_paths, loss_acc, 'train')
        y_valid = CrossValidation.load_results(file_paths, loss_acc, 'valid')
        y_train_mean = mean(y_train, axis=0)
        y_valid_mean = mean(y_valid, axis=0)
        if use_std is True:
            y_train_std = std(y_train, axis=0)
            y_valid_std = std(y_valid, axis=0)
        else:
            y_train_std = 0
            y_valid_std = 0
        n = len(y_train_mean)

        # creating x values
        x = linspace(1, n, n)

        # setting loss_acc plot properties
        if loss_acc == 'loss':
            y_max = ceil(max(y_train_mean + y_train_std))
            y_label = 'loss'
        elif loss_acc == 'acc':
            y_max = 1
            y_label = 'accuracy'
        else:
            raise Exception(
                'loss_acc value "{}" not supported'.format(loss_acc)
            )

        # determining legend
        if use_std is True:
            legend = ['train mean', 'valid mean',
                      r'train mean$\pm$std', r'valid mean$\pm$std']
        else:
            legend = ['train mean', 'valid mean']

        # creating figure
        plt.figure(dpi=100, figsize=(12.8, 7.2), facecolor='w', edgecolor='k')
        plt.style.use('seaborn-whitegrid')

        # plotting train
        plt.plot(x, y_train_mean, '-', color='red')
        if use_std is True:
            plt.fill_between(x,
                             y_train_mean - y_train_std,
                             y_train_mean + y_train_std,
                             color='red', alpha=0.3333)

        # plotting test
        plt.plot(x, y_valid_mean, '-', color='blue')
        if use_std is True:
            plt.fill_between(x,
                             y_valid_mean - y_valid_std,
                             y_valid_mean + y_valid_std,
                             color='blue', alpha=0.3333)

        # manipulating other plot properties
        plt.xlim(1, n)
        plt.ylim(0, y_max)
        plt.legend(labels=legend, fontsize=20, frameon=True)
        plt.xlabel('epochs', fontsize=25)
        plt.ylabel(y_label, fontsize=25)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.title('training set and validation set comparison', fontsize=25)

        # saving figure
        savefig(plot_path)

    @staticmethod
    def tables_manager(file_paths, tables_dir):
        # pre-allocating a new csv row
        csv_loss = []
        csv_acc = []

        # cycling on pickle files
        for idx, file in enumerate(file_paths):
            # getting results from pickle file
            with open(file, 'rb') as f:
                results = load(f)

            # appending new values to csv new line
            csv_loss.append(results['test']['loss'])
            csv_acc.append(results['test']['acc'])

            # printing information
            print(
                'fold', idx + 1,
                '| best epoch: {}'.format(results['best']['idx'] + 1),
                '| with loss: {0:.4f}'.format(csv_loss[-1]),
                '| and acc: {0:.4f}'.format(csv_acc[-1])
            )

        # computing mean (cross-validation)
        csv_loss.append(mean(csv_loss))
        csv_acc.append(mean(csv_acc))

        # printing also this info on mean
        print(
            'mean  ',
            '|', ' ' * 15,
            '| with loss: {0:.4f}'.format(csv_loss[-1]),
            '| and acc: {0:.4f}'.format(csv_acc[-1])
        )

        # creating csv loss and acc paths
        csv_loss_path = join(tables_dir, 'loss.csv')
        csv_acc_path = join(tables_dir, 'acc.csv')

        # # code to remove out of coding phase
        # from os import remove
        # from os.path import exists
        # if exists(csv_loss_path):
        #     remove(csv_loss_path)
        # if exists(csv_acc_path):
        #     remove(csv_acc_path)

        # creating csv files in necessary
        if not exists(csv_loss_path):
            header = ['fold0' + str(x + 1) for x in range(len(file_paths))]
            header.append('mean')
            csv_manager(csv_loss_path, header)
            csv_manager(csv_acc_path, header)

        # adding new row
        csv_manager(csv_loss_path, csv_loss)
        csv_manager(csv_acc_path, csv_acc)
