import itertools
import numpy as np
import logging as log
from copy import deepcopy
from braindecode.datautil.iterators import get_balanced_batches
from hgdecode.signalproc import concatenate_channels, select_trials
from hgdecode.lda import lda_train_scaled, lda_apply


class MultiClassFBCSP(object):
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
                        MultiClassFBCSP.collect_features_for_filter_selection(
                            features,
                            this_filt_per_fb
                        )

                    # make 5 times cross validation...
                    test_accuracy = MultiClassFBCSP.cross_validate_lda(
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
                        MultiClassFBCSP.collect_features_for_filter_selection(
                            features,
                            this_filt_per_fb
                        )
                    # make 5 times cross validation...
                    test_accuracy = MultiClassFBCSP.cross_validate_lda(
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
    def collect_features_for_filter_selection(features,
                                              filters_for_filterband):
        n_filters_per_fb = features[0].X.shape[1] // 2
        n_filterbands = len(features)
        # start with filters of first filterband...
        # then add others all together
        first_features = deepcopy(features[0])
        first_n_filters = filters_for_filterband[0]
        if first_n_filters == 0:
            first_features.X = first_features.X[:, 0:0]
        else:
            first_features.X = first_features.X[:,
                               list(range(first_n_filters))
                               + list(range(-first_n_filters, 0))]

        all_features = first_features
        for i in range(1, n_filterbands):
            this_n_filters = min(n_filters_per_fb, filters_for_filterband[i])
            if this_n_filters > 0:
                next_features = deepcopy(features[i])
                if this_n_filters == 0:
                    next_features.X = next_features.X[0:0]
                else:
                    next_features.X = next_features.X[:,
                                      list(range(this_n_filters)) +
                                      list(range(-this_n_filters, 0))]
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
