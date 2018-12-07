from numpy import empty, mean, array
from logging import getLogger
from hgdecode.lda import lda_apply
from hgdecode.lda import lda_train_scaled
from hgdecode.signalproc import bandpass_mne
from hgdecode.signalproc import select_trials
from hgdecode.signalproc import calculate_csp
from hgdecode.signalproc import select_classes
from hgdecode.signalproc import apply_csp_var_log
from braindecode.datautil.trial_segment import \
    create_signal_target_from_raw_mne

log = getLogger(__name__)


class BinaryFBCSP(object):
    """
    # TODO: a description of this class
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
        log.info("\n")
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
