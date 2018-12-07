# General modules
from numpy import arange
from numpy import setdiff1d
from numpy import int as npint
from logging import getLogger
from itertools import combinations
from numpy.random import RandomState
from hgdecode.utils import print_manager
from hgdecode.classes import FilterBank
from braindecode.datautil.iterators import get_balanced_batches

# Deep Learning
from hgdecode.models import import_model

# Machine Learning
from hgdecode.binary import BinaryFBCSP
from hgdecode.multiclass import MultiClassFBCSP
from hgdecode.multiclass import MultiClassWeightedVoting

log = getLogger(__name__)


class FBCSPExperiment(object):
    """
        A Filter Bank Common Spatial Patterns Experiment.

        Parameters
        ----------
        cnt : RawArray
            The continuous train recordings with events in info['events']
        clean_trial_mask : bool array
            Bool array containing information about valid/invalid trials
        name_to_start_codes: dict
            Dictionary mapping class names to marker numbers, e.g.
            {'1 - Correct': [31], '2 - Error': [32]}
        epoch_ival_ms : sequence of 2 floats
            The start and end of the trial in milliseconds with respect to
            the markers.
        min_freq : int or list or tuple
            The minimum frequency/ies of the filterbank/s.
        max_freq : int or list or tuple
            The maximum frequency/ies of the filterbank/s.
        window : int or list or tuple
            Bandwidths of filters in filterbank/s.
        overlap : int or list or tuple
            Overlap frequencies between filters in filterbank/s.
        filt_order : int
            The filter order of the butterworth filter which computes the
            filterbands.
        n_folds : int
            How many folds. Also determines size of the test fold, e.g.
            5 folds imply the test fold has 20% of the original data.
        n_top_bottom_csp_filters : int or None
            Number of top and bottom CSP filters to select from all computed
            filters. Top and bottom refers to CSP filters sorted by their
            eigenvalues. So a value of 3 here will lead to 6(!) filters.
            None means all filters.
        n_selected_filterbands : int or None
            Number of filterbands to select for the filterbank.
            Will be selected by the highest training accuracies.
            None means all filterbands.
        n_selected_features : int or None
            Number of features to select for the filterbank.
            Will be selected by an internal cross validation across feature
            subsets.
            None means all features.
        forward_steps : int
            Number of forward steps to make in the feature selection,
            before the next backward step.
        backward_steps : int
            Number of backward steps to make in the feature selection,
            before the next forward step.
        stop_when_no_improvement: bool
            Whether to stop the feature selection if the internal cross
            validation accuracy could not be improved after an epoch finished
            (epoch=given number of forward and backward steps).
            False implies always run until wanted number of features.
        shuffle: bool
            Whether to shuffle the clean trials before splitting them into
            folds. False implies folds are time-blocks, True implies folds are
            random mixes of trials of the entire file.
    """

    def __init__(self,
                 # signal-related inputs
                 cnt,
                 clean_trial_mask,
                 name_to_start_codes,
                 name_to_stop_codes=None,
                 epoch_ival_ms=(-500, 4000),

                 # bank filter-related inputs
                 min_freq=0,
                 max_freq=12,
                 window=6,
                 overlap=3,
                 filt_order=3,

                 # machine learning-related inputs
                 n_folds=5,
                 n_top_bottom_csp_filters=None,
                 n_selected_filterbands=None,
                 n_selected_features=None,
                 forward_steps=2,
                 backward_steps=1,
                 stop_when_no_improvement=False,
                 shuffle=False,
                 average_trial_covariance=True):
        # signal-related inputs
        self.cnt = cnt
        self.clean_trial_mask = clean_trial_mask
        self.epoch_ival_ms = epoch_ival_ms
        self.name_to_start_codes = name_to_start_codes
        self.name_to_stop_codes = name_to_stop_codes

        # bank filter-related inputs
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.window = window
        self.overlap = overlap
        self.filt_order = filt_order

        # machine learning-related inputs
        self.n_folds = n_folds
        self.n_top_bottom_csp_filters = n_top_bottom_csp_filters
        self.n_selected_filterbands = n_selected_filterbands
        self.n_selected_features = n_selected_features
        self.forward_steps = forward_steps
        self.backward_steps = backward_steps
        self.stop_when_no_improvement = stop_when_no_improvement
        self.shuffle = shuffle
        self.average_trial_covariance = average_trial_covariance

        # other fundamental properties (they will be filled in further
        # computational steps)
        self.filterbank_csp = None
        self.class_pairs = None
        self.folds = None
        self.binary_csp = None
        self.filterbands = None
        self.multi_class = None

        # computing other properties for further computation
        self.n_classes = len(self.name_to_start_codes)
        self.class_pairs = list(combinations(range(self.n_classes), 2))
        self.n_trials = self.clean_trial_mask.astype(npint).sum()

    def create_filter_bank(self):
        self.filterbands = FilterBank(
            min_freq=self.min_freq,
            max_freq=self.max_freq,
            window=self.window,
            overlap=self.overlap
        )

    def create_folds(self):
        # getting pseudo-random folds
        folds = get_balanced_batches(
            n_trials=self.n_trials,
            rng=RandomState(1234),
            shuffle=self.shuffle,
            n_batches=self.n_folds
        )

        # remapping to original indices in unclean set(!)
        # train is everything except fold
        # test is fold indexes
        self.folds = [
            {
                'train': setdiff1d(arange(self.n_trials), fold),
                'test': fold
            }
            for fold in folds
        ]

    def run(self):
        # printing routine start
        print_manager(
            'TRAINING ROUTINE',
            'double-dashed',
        )

        # creating filter bank
        print_manager('Creating filter bank...')
        self.create_filter_bank()
        print_manager('DONE!!', bottom_return=1)

        # creating folds
        print_manager('Creating folds...')
        self.create_folds()
        print_manager('DONE!!', bottom_return=1)

        # printing the start
        log.info("Running Training...")

        # creating a BinaryFBCSP instance
        self.binary_csp = BinaryFBCSP(
            cnt=self.cnt,
            clean_trial_mask=self.clean_trial_mask,
            filterbands=self.filterbands,
            filt_order=self.filt_order,
            folds=self.folds,
            class_pairs=self.class_pairs,
            epoch_ival_ms=self.epoch_ival_ms,
            n_filters=self.n_top_bottom_csp_filters,
            marker_def=self.name_to_start_codes,
            name_to_stop_codes=self.name_to_stop_codes,
            average_trial_covariance=self.average_trial_covariance
        )

        # running it
        self.binary_csp.run()

        # TODO: understand filterbankCSP

        # at the very end of the binary CSP experiment...
        log.info("Filterbank...")
        self.filterbank_csp = MultiClassFBCSP(
            binary_csp=self.binary_csp,
            n_features=self.n_selected_features,
            n_filterbands=self.n_selected_filterbands,
            forward_steps=self.forward_steps,
            backward_steps=self.backward_steps,
            stop_when_no_improvement=self.stop_when_no_improvement
        )

        # running it
        self.filterbank_csp.run()

        # TODO: understand MultiClassWeightedVoting as well

        # and finally multiclass...
        log.info("Multiclass...")
        self.multi_class = MultiClassWeightedVoting(
            train_labels=self.binary_csp.train_labels_full_fold,
            test_labels=self.binary_csp.test_labels_full_fold,
            train_preds=self.filterbank_csp.train_pred_full_fold,
            test_preds=self.filterbank_csp.test_pred_full_fold,
            class_pairs=self.class_pairs)
        self.multi_class.run()


class DLExperiment(object):
    """
    # TODO: a description for this class
    """

    def __init__(self,
                 dataset,
                 model_name,
                 results_dir,
                 batch_size=128,
                 epochs=6,
                 verbose=False,
                 validation_frac=0.2,
                 loss='categorical_crossentropy',
                 optimizer='Adam',
                 metrics='None',
                 shuffle='False'):
        # resolving PEP8 requests on mutable input arguments
        if metrics is 'None':
            metrics = ['accuracy']

        # non-default inputs
        self.dataset = dataset
        self.model_name = model_name
        self.results_dir = results_dir

        # hyperparameters
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.validation_frac = validation_frac
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        self.shuffle = shuffle

        # importing model
        self.model = import_model(self)

    def __repr__(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
