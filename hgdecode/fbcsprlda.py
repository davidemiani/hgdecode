import itertools
import numpy as np
import logging as log
from copy import deepcopy
from numpy import empty, mean, array
from hgdecode.lda import lda_apply
from hgdecode.lda import lda_train_scaled
from hgdecode.signalproc import bandpass_mne
from hgdecode.signalproc import select_trials
from hgdecode.signalproc import calculate_csp
from hgdecode.signalproc import select_classes
from hgdecode.signalproc import apply_csp_var_log
from hgdecode.signalproc import concatenate_channels
from braindecode.datautil.iterators import get_balanced_batches
from braindecode.datautil.trial_segment import \
    create_signal_target_from_raw_mne


class BinaryFBCSP(object):
    """
    # TODO: a description for this class
    """

    def __init__(self,
                 cnt,
                 clean_trial_mask,
                 filterbands,
                 filt_order,
                 folds,
                 class_pairs,
                 epoch_ival_ms,
                 n_filters,
                 marker_def,
                 name_to_stop_codes=None,
                 average_trial_covariance=False):
        # cnt and signal parameters
        self.cnt = cnt
        self.clean_trial_mask = clean_trial_mask
        self.epoch_ival_ms = epoch_ival_ms
        self.marker_def = marker_def
        self.name_to_stop_codes = name_to_stop_codes

        # filter bank parameters
        self.filterbands = filterbands
        self.filt_order = filt_order

        # machine learning parameters
        self.folds = folds
        self.n_filters = n_filters
        self.class_pairs = class_pairs
        self.average_trial_covariance = average_trial_covariance

        # getting result shape
        n_filterbands = len(self.filterbands)
        n_folds = len(self.folds)
        n_class_pairs = len(self.class_pairs)
        result_shape = (n_filterbands, n_folds, n_class_pairs)

        # creating result related properties and pre-allocating them
        self.filters = empty(result_shape, dtype=object)
        self.patterns = empty(result_shape, dtype=object)
        self.variances = empty(result_shape, dtype=object)
        self.train_feature = empty(result_shape, dtype=object)
        self.test_feature = empty(result_shape, dtype=object)
        self.train_feature_full_fold = empty(result_shape, dtype=object)
        self.test_feature_full_fold = empty(result_shape, dtype=object)
        self.clf = empty(result_shape, dtype=object)
        self.train_accuracy = empty(result_shape, dtype=object)
        self.test_accuracy = empty(result_shape, dtype=object)
        self.train_labels_full_fold = empty(len(self.folds), dtype=object)
        self.test_labels_full_fold = empty(len(self.folds), dtype=object)
        self.train_labels = empty(
            (len(self.folds), len(self.class_pairs)),
            dtype=object)
        self.test_labels = empty(
            (len(self.folds), len(self.class_pairs)),
            dtype=object)

    def run(self):
        # %% CYCLING ON FILTERS IN FILTERBANK
        # %%
        # just for me: enumerate is a really powerful built-in python
        # function that allows you to loop over something and have an
        # automatic counter. In this case, bp_nr is the counter,
        # then filt_band is the default exit for the method getitem for
        # filterbands class.
        for bp_nr, filt_band in enumerate(self.filterbands):
            # printing filter information
            self.print_filter(bp_nr)

            # bandpassing all the cnt RawArray with the current filter
            bandpassed_cnt = bandpass_mne(
                self.cnt,
                filt_band[0],
                filt_band[1],
                filt_order=self.filt_order
            )

            # epoching: from cnt data to epoched data
            epo = create_signal_target_from_raw_mne(
                bandpassed_cnt,
                name_to_start_codes=self.marker_def,
                epoch_ival_ms=self.epoch_ival_ms,
                name_to_stop_codes=self.name_to_stop_codes
            )

            # cleaning epoched data with clean_trial_mask (finally)
            epo.X = epo.X[self.clean_trial_mask]
            epo.y = epo.y[self.clean_trial_mask]

            # %% CYCLING ON FOLDS
            # %%
            for fold_nr in range(len(self.folds)):
                # self.run_fold(epo, bp_nr, fold_nr)

                # printing fold information
                self.print_fold_nr(fold_nr)

                # getting information on current fold
                train_test = self.folds[fold_nr]

                # getting train and test indexes
                train_ind = train_test['train']
                test_ind = train_test['test']

                # getting train data from train indexes
                epo_train = select_trials(epo, train_ind)

                # getting test data from test indexes
                epo_test = select_trials(epo, test_ind)

                # logging info on train
                log.info("#Train trials: {:4d}".format(len(epo_train.X)))

                # logging info on test
                log.info("#Test trials : {:4d}".format(len(epo_test.X)))

                # setting train labels of this fold
                self.train_labels_full_fold[fold_nr] = epo_train.y

                # setting test labels of this fold
                self.test_labels_full_fold[fold_nr] = epo_test.y

                # %% CYCLING ON ALL POSSIBLE CLASS PAIRS
                # %%
                for pair_nr in range(len(self.class_pairs)):
                    # getting class pair from index (pair_nr)
                    class_pair = self.class_pairs[pair_nr]

                    # printing class pair information
                    self.print_class_pair(class_pair)

                    # getting train trials only for current two classes
                    epo_train_pair = select_classes(epo_train, class_pair)

                    # getting test trials only for current two classes
                    epo_test_pair = select_classes(epo_test, class_pair)

                    # saving train labels for this two classes
                    self.train_labels[fold_nr][pair_nr] = epo_train_pair.y

                    # saving test labels for this two classes
                    self.test_labels[fold_nr][pair_nr] = epo_test_pair.y

                    # %% COMPUTING CSP
                    # %%
                    filters, patterns, variances = calculate_csp(
                        epo_train_pair,
                        average_trial_covariance=self.average_trial_covariance
                    )

                    # %% FEATURE EXTRACTION
                    # %%
                    # choosing how many spacial filter to apply;
                    # if no spacial filter number specified...
                    if self.n_filters is None:
                        # ...taking all columns, else...
                        columns = list(range(len(filters)))
                    else:
                        # ...take topmost and bottommost filters;
                        # e.g. for n_filters=3 we are going to pick:
                        # 0, 1, 2, -3, -2, -1
                        columns = (list(range(0, self.n_filters)) +
                                   list(range(-self.n_filters, 0)))

                    # feature extraction on train
                    train_feature = apply_csp_var_log(
                        epo_train_pair,
                        filters,
                        columns
                    )

                    # feature extraction on test
                    test_feature = apply_csp_var_log(
                        epo_test_pair,
                        filters,
                        columns
                    )

                    # %% COMPUTING LDA USING TRAIN FEATURES
                    # %%
                    # clf is a 1x2 tuple where:
                    #    * clf[0] is hyperplane parameters
                    #    * clf[1] is hyperplane bias
                    # with clf, you can recreate the n-dimensional
                    # hyperplane that splits class space, so you can
                    # classify your fbcsp extracted features.
                    clf = lda_train_scaled(train_feature, shrink=True)

                    # %% APPLYING LDA ON TRAIN
                    # %%
                    # applying LDA
                    train_out = lda_apply(train_feature, clf)

                    # getting true/false labels instead of class labels
                    #    for example, if you have:
                    #    train_feature.y --> [1, 3, 3, 1]
                    #    class_pair --> [1, 3]
                    #    so you will have:
                    #    true_0_1_labels_train = [False, True, True, False]
                    true_0_1_labels_train = train_feature.y == class_pair[1]

                    # if predicted output grater than 0 True, False instead
                    predicted_train = train_out >= 0

                    # computing accuracy
                    #    if mean has a boolean array as input, it will
                    #    compute number of True elements divided by total
                    #    number of elements, so the accuracy
                    train_accuracy = mean(
                        true_0_1_labels_train == predicted_train
                    )

                    # %% APPLYING LDA ON TEST
                    # %%
                    # same procedure
                    test_out = lda_apply(test_feature, clf)
                    true_0_1_labels_test = test_feature.y == class_pair[1]
                    predicted_test = test_out >= 0
                    test_accuracy = mean(
                        true_0_1_labels_test == predicted_test
                    )

                    # %% FEATURE COMPUTATION FOR FULL FOLD
                    # %% (FOR LATER MULTICLASS)
                    # here we use csp computed only for this pair of classes
                    # to compute feature for all the current fold
                    # train here
                    train_feature_full_fold = apply_csp_var_log(
                        epo_train,
                        filters,
                        columns
                    )

                    # test here
                    test_feature_full_fold = apply_csp_var_log(
                        epo_test,
                        filters,
                        columns
                    )

                    # %% STORE RESULTS
                    # %%
                    # only store used patterns filters variances
                    # to save memory space on disk
                    self.store_results(
                        bp_nr,
                        fold_nr,
                        pair_nr,
                        filters[:, columns],
                        patterns[:, columns],
                        variances[columns],
                        train_feature,
                        test_feature,
                        train_feature_full_fold,
                        test_feature_full_fold,
                        clf,
                        train_accuracy,
                        test_accuracy
                    )

                    # printing the end of this super-nested cycle
                    self.print_results(bp_nr, fold_nr, pair_nr)

            # printing a blank line to divide filters
            print()

    def store_results(self,
                      bp_nr,
                      fold_nr,
                      pair_nr,
                      filters,
                      patterns,
                      variances,
                      train_feature,
                      test_feature,
                      train_feature_full_fold,
                      test_feature_full_fold,
                      clf,
                      train_accuracy,
                      test_accuracy):
        """ Store all supplied arguments to this objects dict, at the correct
        indices for filterband / fold / class_pair."""
        local_vars = locals()
        del local_vars['self']
        del local_vars['bp_nr']
        del local_vars['fold_nr']
        del local_vars['pair_nr']
        for var in local_vars:
            self.__dict__[var][bp_nr, fold_nr, pair_nr] = local_vars[var]

    def print_filter(self, bp_nr):
        # distinguish filter blocks by empty line
        log.info(
            "Filter {:d}/{:d}, {:4.2f} to {:4.2f} Hz".format(
                bp_nr + 1,
                len(self.filterbands),
                *self.filterbands[bp_nr])
        )

    @staticmethod
    def print_fold_nr(fold_nr):
        log.info("Fold Nr: {:d}".format(fold_nr + 1))

    @staticmethod
    def print_class_pair(class_pair):
        class_pair_plus_one = (array(class_pair) + 1).tolist()
        log.info("Class {:d} vs {:d}".format(*class_pair_plus_one))

    def print_results(self, bp_nr, fold_nr, pair_nr):
        log.info("Train: {:4.2f}%".format(
            self.train_accuracy[bp_nr, fold_nr, pair_nr] * 100))
        log.info("Test:  {:4.2f}%".format(
            self.test_accuracy[bp_nr, fold_nr, pair_nr] * 100))


class FBCSP(object):
    """
    # TODO: a description for this class
    """

    def __init__(self,
                 binary_csp,
                 n_features=None,
                 n_filterbands=None,
                 forward_steps=2,
                 backward_steps=1,
                 stop_when_no_improvement=False):
        # copying inputs
        self.binary_csp = binary_csp
        self.n_features = n_features
        self.n_filterbands = n_filterbands
        self.forward_steps = forward_steps
        self.backward_steps = backward_steps
        self.stop_when_no_improvement = stop_when_no_improvement

        # pre-allocating other properties
        self.train_feature = None
        self.train_feature_full_fold = None
        self.test_feature = None
        self.test_feature_full_fold = None
        self.selected_filter_inds = None
        self.selected_filters_per_filterband = None
        self.selected_features = None
        self.clf = None
        self.train_accuracy = None
        self.test_accuracy = None
        self.train_pred_full_fold = None
        self.test_pred_full_fold = None

    def run(self):
        self.select_filterbands()
        if self.n_features is not None:
            log.info("Run feature selection...")
            self.collect_best_features()
            log.info("Done.")
        else:
            self.collect_features()
        self.train_classifiers()
        self.predict_outputs()

    def select_filterbands(self):
        n_all_filterbands = len(self.binary_csp.filterbands)
        if self.n_filterbands is None:
            self.selected_filter_inds = list(range(n_all_filterbands))
        else:
            # Select the filterbands with the highest mean accuracy on the
            # training sets
            mean_accs = np.mean(self.binary_csp.train_accuracy, axis=(1, 2))
            best_filters = np.argsort(mean_accs)[::-1][:self.n_filterbands]
            self.selected_filter_inds = best_filters

    def collect_features(self):
        n_folds = len(self.binary_csp.folds)
        n_class_pairs = len(self.binary_csp.class_pairs)
        result_shape = (n_folds, n_class_pairs)
        self.train_feature = np.empty(result_shape, dtype=object)
        self.train_feature_full_fold = np.empty(result_shape, dtype=object)
        self.test_feature = np.empty(result_shape, dtype=object)
        self.test_feature_full_fold = np.empty(result_shape, dtype=object)

        bcsp = self.binary_csp  # just to make code shorter
        filter_inds = self.selected_filter_inds
        for fold_i in range(n_folds):
            for class_i in range(n_class_pairs):
                self.train_feature[fold_i, class_i] = concatenate_channels(
                    bcsp.train_feature[filter_inds, fold_i, class_i])
                self.train_feature_full_fold[fold_i, class_i] = (
                    concatenate_channels(
                        bcsp.train_feature_full_fold[
                            filter_inds, fold_i, class_i]))
                self.test_feature[fold_i, class_i] = concatenate_channels(
                    bcsp.test_feature[filter_inds, fold_i, class_i]
                )
                self.test_feature_full_fold[fold_i, class_i] = (
                    concatenate_channels(
                        bcsp.test_feature_full_fold[
                            filter_inds, fold_i, class_i]
                    ))

    def collect_best_features(self):
        """ Selects features filterwise per filterband, starting with no
        features, then selecting the best filterpair from the bestfilterband
        (measured on internal train/test split)"""
        bincsp = self.binary_csp  # just to make code shorter

        # getting dimension for feature arrays
        n_folds = len(self.binary_csp.folds)
        n_class_pairs = len(self.binary_csp.class_pairs)
        result_shape = (n_folds, n_class_pairs)

        # initializing feature array for this classes
        self.train_feature = np.empty(result_shape, dtype=object)
        self.train_feature_full_fold = np.empty(result_shape, dtype=object)
        self.test_feature = np.empty(result_shape, dtype=object)
        self.test_feature_full_fold = np.empty(result_shape, dtype=object)
        self.selected_filters_per_filterband = np.empty(result_shape,
                                                        dtype=object)
        # outer cycle on folds
        for fold_i in range(n_folds):
            # inner cycle on pairs
            for class_pair_i in range(n_class_pairs):
                # saving bincsp features locally (prevent ram to modify values)
                bin_csp_train_features = deepcopy(
                    bincsp.train_feature[
                        self.selected_filter_inds, fold_i, class_pair_i
                    ]
                )
                bin_csp_train_features_full_fold = deepcopy(
                    bincsp.train_feature_full_fold[
                        self.selected_filter_inds,
                        fold_i, class_pair_i
                    ]
                )
                bin_csp_test_features = deepcopy(
                    bincsp.test_feature[
                        self.selected_filter_inds,
                        fold_i,
                        class_pair_i
                    ]
                )
                bin_csp_test_features_full_fold = deepcopy(
                    bincsp.test_feature_full_fold[
                        self.selected_filter_inds, fold_i, class_pair_i
                    ]
                )

                # selecting best filters
                selected_filters_per_filt = \
                    self.select_best_filters_best_filterbands(
                        bin_csp_train_features,
                        max_features=self.n_features,
                        forward_steps=self.forward_steps,
                        backward_steps=self.backward_steps,
                        stop_when_no_improvement=self.stop_when_no_improvement
                    )

                # collecting train features
                self.train_feature[fold_i, class_pair_i] = \
                    self.collect_features_for_filter_selection(
                        bin_csp_train_features,
                        selected_filters_per_filt
                    )

                # collecting train features full fold
                self.train_feature_full_fold[fold_i, class_pair_i] = \
                    self.collect_features_for_filter_selection(
                        bin_csp_train_features_full_fold,
                        selected_filters_per_filt
                    )

                # collecting test features
                self.test_feature[fold_i, class_pair_i] = \
                    self.collect_features_for_filter_selection(
                        bin_csp_test_features,
                        selected_filters_per_filt
                    )

                # collecting test features full fold
                self.test_feature_full_fold[fold_i, class_pair_i] = \
                    self.collect_features_for_filter_selection(
                        bin_csp_test_features_full_fold,
                        selected_filters_per_filt
                    )

                # saving also the filters selected for this fold and pair
                self.selected_filters_per_filterband[fold_i, class_pair_i] = \
                    selected_filters_per_filt

    @staticmethod
    def select_best_filters_best_filterbands(features,
                                             max_features,
                                             forward_steps,
                                             backward_steps,
                                             stop_when_no_improvement):
        n_filterbands = len(features)
        n_filters_per_fb = features[0].X.shape[1] / 2
        selected_filters_per_band = [0] * n_filterbands
        best_selected_filters_per_filterband = None
        last_best_accuracy = -1

        # Run until no improvement or max features reached
        selection_finished = False
        while not selection_finished:
            for _ in range(forward_steps):
                best_accuracy = -1

                # let's try always taking a feature in each iteration
                for filt_i in range(n_filterbands):
                    this_filt_per_fb = deepcopy(selected_filters_per_band)
                    if this_filt_per_fb[filt_i] == n_filters_per_fb:
                        continue
                    this_filt_per_fb[filt_i] = this_filt_per_fb[filt_i] + 1
                    all_features = \
                        FBCSP.collect_features_for_filter_selection(
                            features,
                            this_filt_per_fb
                        )

                    # make 5 times cross validation...
                    test_accuracy = FBCSP.cross_validate_lda(
                        all_features
                    )

                    if test_accuracy > best_accuracy:
                        best_accuracy = test_accuracy
                        best_selected_filters_per_filterband = this_filt_per_fb

                selected_filters_per_band = \
                    best_selected_filters_per_filterband

            for _ in range(backward_steps):
                best_accuracy = -1
                # let's try always taking a feature in each iteration
                for filt_i in range(n_filterbands):
                    this_filt_per_fb = deepcopy(selected_filters_per_band)
                    if this_filt_per_fb[filt_i] == 0:
                        continue
                    this_filt_per_fb[filt_i] = this_filt_per_fb[filt_i] - 1
                    all_features = \
                        FBCSP.collect_features_for_filter_selection(
                            features,
                            this_filt_per_fb
                        )
                    # make 5 times cross validation...
                    test_accuracy = FBCSP.cross_validate_lda(
                        all_features
                    )
                    if test_accuracy > best_accuracy:
                        best_accuracy = test_accuracy
                        best_selected_filters_per_filterband = this_filt_per_fb
                selected_filters_per_band = \
                    best_selected_filters_per_filterband

            selection_finished = 2 * np.sum(
                selected_filters_per_band) >= max_features
            if stop_when_no_improvement:
                # there was no improvement if accuracy did not increase...
                selection_finished = (selection_finished
                                      or best_accuracy <= last_best_accuracy)
            last_best_accuracy = best_accuracy
        return selected_filters_per_band

    @staticmethod
    def collect_features_for_filter_selection(
            features,
            filters_for_filterband
    ):
        n_filters_per_fb = features[0].X.shape[1] // 2
        n_filterbands = len(features)
        # start with filters of first filterband...
        # then add others all together
        first_features = deepcopy(features[0])
        first_n_filters = filters_for_filterband[0]
        if first_n_filters == 0:
            first_features.X = first_features.X[:, 0:0]
        else:
            first_features.X = \
                first_features.X[
                :,
                list(range(first_n_filters)) +
                list(range(-first_n_filters, 0))
                ]

        all_features = first_features
        for i in range(1, n_filterbands):
            this_n_filters = min(n_filters_per_fb, filters_for_filterband[i])
            if this_n_filters > 0:
                next_features = deepcopy(features[i])
                if this_n_filters == 0:
                    next_features.X = next_features.X[0:0]
                else:
                    next_features.X = \
                        next_features.X[
                        :,
                        list(range(this_n_filters)) +
                        list(range(-this_n_filters, 0))
                        ]
                all_features = concatenate_channels(
                    (all_features, next_features))
        return all_features

    @staticmethod
    def cross_validate_lda(features):
        n_trials = features.X.shape[0]
        folds = get_balanced_batches(n_trials, rng=None, shuffle=False,
                                     n_batches=5)
        # make to train-test splits, fold is test part..
        folds = [(np.setdiff1d(np.arange(n_trials), fold),
                  fold) for fold in folds]
        test_accuracies = []
        for train_inds, test_inds in folds:
            train_features = select_trials(features, train_inds)
            test_features = select_trials(features, test_inds)
            clf = lda_train_scaled(train_features, shrink=True)
            test_out = lda_apply(test_features, clf)

            higher_class = np.max(test_features.y)
            true_0_1_labels_test = test_features.y == higher_class

            predicted_test = test_out >= 0
            test_accuracy = np.mean(true_0_1_labels_test == predicted_test)
            test_accuracies.append(test_accuracy)
        return np.mean(test_accuracies)

    def train_classifiers(self):
        n_folds = len(self.binary_csp.folds)
        n_class_pairs = len(self.binary_csp.class_pairs)
        self.clf = np.empty((n_folds, n_class_pairs),
                            dtype=object)
        for fold_i in range(n_folds):
            for class_i in range(n_class_pairs):
                train_feature = self.train_feature[fold_i, class_i]
                clf = lda_train_scaled(train_feature, shrink=True)
                self.clf[fold_i, class_i] = clf

    def predict_outputs(self):
        n_folds = len(self.binary_csp.folds)
        n_class_pairs = len(self.binary_csp.class_pairs)
        result_shape = (n_folds, n_class_pairs)
        self.train_accuracy = np.empty(result_shape, dtype=float)
        self.test_accuracy = np.empty(result_shape, dtype=float)
        self.train_pred_full_fold = np.empty(result_shape, dtype=object)
        self.test_pred_full_fold = np.empty(result_shape, dtype=object)
        for fold_i in range(n_folds):
            log.info("Fold Nr: {:d}".format(fold_i + 1))
            for class_i, class_pair in enumerate(self.binary_csp.class_pairs):
                clf = self.clf[fold_i, class_i]
                class_pair_plus_one = (np.array(class_pair) + 1).tolist()
                log.info("Class {:d} vs {:d}".format(*class_pair_plus_one))
                train_feature = self.train_feature[fold_i, class_i]
                train_out = lda_apply(train_feature, clf)
                true_0_1_labels_train = train_feature.y == class_pair[1]
                predicted_train = train_out >= 0
                # remove xarray wrapper with float( ...
                train_accuracy = float(np.mean(true_0_1_labels_train
                                               == predicted_train))
                self.train_accuracy[fold_i, class_i] = train_accuracy

                test_feature = self.test_feature[fold_i, class_i]
                test_out = lda_apply(test_feature, clf)
                true_0_1_labels_test = test_feature.y == class_pair[1]
                predicted_test = test_out >= 0
                test_accuracy = float(np.mean(true_0_1_labels_test
                                              == predicted_test))

                self.test_accuracy[fold_i, class_i] = test_accuracy

                train_feature_full_fold = self.train_feature_full_fold[fold_i,
                                                                       class_i]
                train_out_full_fold = lda_apply(train_feature_full_fold, clf)
                self.train_pred_full_fold[
                    fold_i, class_i] = train_out_full_fold
                test_feature_full_fold = self.test_feature_full_fold[fold_i,
                                                                     class_i]
                test_out_full_fold = lda_apply(test_feature_full_fold, clf)
                self.test_pred_full_fold[fold_i, class_i] = test_out_full_fold

                log.info("Train: {:4.2f}%".format(train_accuracy * 100))
                log.info("Test:  {:4.2f}%".format(test_accuracy * 100))


class MultiClassWeightedVoting(object):
    """
    # TODO: a description for this class
    """

    def __init__(self,
                 train_labels,
                 test_labels,
                 train_preds,
                 test_preds,
                 class_pairs):
        # copying input parameters
        self.train_labels = train_labels
        self.test_labels = test_labels
        self.train_preds = train_preds
        self.test_preds = test_preds
        self.class_pairs = class_pairs

        # pre-allocating other useful properties
        self.train_class_sums = None
        self.test_class_sums = None
        self.train_predicted_labels = None
        self.test_predicted_labels = None
        self.train_accuracy = None
        self.test_accuracy = None

    def run(self):
        # determine number of classes by number of unique classes
        # appearing in class pairs
        n_classes = len(np.unique(list(itertools.chain(*self.class_pairs))))
        n_folds = len(self.train_labels)
        self.train_class_sums = np.empty(n_folds, dtype=object)
        self.test_class_sums = np.empty(n_folds, dtype=object)
        self.train_predicted_labels = np.empty(n_folds, dtype=object)
        self.test_predicted_labels = np.empty(n_folds, dtype=object)
        self.train_accuracy = np.ones(n_folds) * np.nan
        self.test_accuracy = np.ones(n_folds) * np.nan
        for fold_nr in range(n_folds):
            log.info("Fold Nr: {:d}".format(fold_nr + 1))
            train_labels = self.train_labels[fold_nr]
            train_preds = self.train_preds[fold_nr]
            train_class_sums = np.zeros((len(train_labels), n_classes))

            test_labels = self.test_labels[fold_nr]
            test_preds = self.test_preds[fold_nr]
            test_class_sums = np.zeros((len(test_labels), n_classes))
            for pair_i, class_pair in enumerate(self.class_pairs):
                this_train_preds = train_preds[pair_i]
                assert len(this_train_preds) == len(train_labels)
                train_class_sums[:, class_pair[0]] -= this_train_preds
                train_class_sums[:, class_pair[1]] += this_train_preds
                this_test_preds = test_preds[pair_i]
                assert len(this_test_preds) == len(test_labels)
                test_class_sums[:, class_pair[0]] -= this_test_preds
                test_class_sums[:, class_pair[1]] += this_test_preds

            self.train_class_sums[fold_nr] = train_class_sums
            self.test_class_sums[fold_nr] = test_class_sums
            train_predicted_labels = np.argmax(train_class_sums, axis=1)
            test_predicted_labels = np.argmax(test_class_sums, axis=1)
            self.train_predicted_labels[fold_nr] = train_predicted_labels
            self.test_predicted_labels[fold_nr] = test_predicted_labels
            train_accuracy = (np.sum(train_predicted_labels == train_labels) /
                              float(len(train_labels)))
            self.train_accuracy[fold_nr] = train_accuracy
            test_accuracy = (np.sum(test_predicted_labels == test_labels) /
                             float(len(test_labels)))
            self.test_accuracy[fold_nr] = test_accuracy
            log.info("Train: {:4.2f}%".format(train_accuracy * 100))
            log.info("Test:  {:4.2f}%".format(test_accuracy * 100))
