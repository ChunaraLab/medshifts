'''
From http://zachmoshe.com/2017/04/03/pickling-keras-models.html
'''

from collections import namedtuple
import numpy as np
import scipy.special
import six
from six.moves import range
# from fairlearn.metrics import group_recall_score, group_specificity_score, group_accuracy_score, group_mean_prediction
from fairlearn.metrics import MetricFrame
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference, false_negative_rate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

import types
import tempfile
import keras.models


Metrics = namedtuple('Metrics',('brier', 'mae', 'logloss',
                                'auc', 'acc', 'smr', 'citl', 'cs', 'ece',
                                'eo', 'dp', 'fnr', 'citl_diff', 'cs_diff'))


class MetricResults(object):
  def __init__(self, num_shifts, num_sample_choices, num_runs):
    self.results_train = dict()
    self.results_test = dict()
    self.results_val = dict()
    self.num_shifts = num_shifts
    self.num_sample_choices = num_sample_choices
    self.num_runs = num_runs

  def add(self, d_train, d_test, d_val, shift_id, sample_id, run_id):

    d_train_ = Metrics(brier=d_train['brier'], mae=d_train['mae'], logloss=d_train['logloss'],
                auc=d_train['auc'], acc=d_train['acc'],
                smr=d_train['smr'], citl=d_train['citl'], cs=d_train['cs'], ece=d_train['ece'],
                eo=d_train['eo'], dp=d_train['dp'], fnr=d_train['fnr'],
                citl_diff=d_train['citl_diff'], cs_diff=d_train['cs_diff'])

    d_test_ = Metrics(brier=d_test['brier'], mae=d_test['mae'], logloss=d_test['logloss'],
                auc=d_test['auc'], acc=d_test['acc'],
                smr=d_test['smr'], citl=d_test['citl'], cs=d_test['cs'], ece=d_test['ece'],
                eo=d_test['eo'], dp=d_test['dp'], fnr=d_test['fnr'],
                citl_diff=d_test['citl_diff'], cs_diff=d_test['cs_diff'])

    d_val_ = Metrics(brier=d_val['brier'], mae=d_val['mae'], logloss=d_val['logloss'],
                auc=d_val['auc'], acc=d_val['acc'],
                smr=d_val['smr'], citl=d_val['citl'], cs=d_val['cs'], ece=d_val['ece'],
                eo=d_val['eo'], dp=d_val['dp'], fnr=d_val['fnr'],
                citl_diff=d_val['citl_diff'], cs_diff=d_val['cs_diff'])

    self.results_train[shift_id, sample_id, run_id] = d_train_
    self.results_test[shift_id, sample_id, run_id] = d_test_
    self.results_val[shift_id, sample_id, run_id] = d_val_


def get_metrics_array(results):
  num_shifts = results.num_shifts
  num_sample_choices = results.num_sample_choices
  num_runs = results.num_runs
  metric_names = ['te_val_diff_auc','te_val_diff_smr','eo','dp','te_val_diff_brier',
                  'citl','cs','fnr','citl_diff','cs_diff']
  results_array = np.ones((num_sample_choices, num_shifts, num_runs, 10)) * (-1) # 0-auc, 1-smr, 2-eo, 3-dp
  for shift_id, sample_id, run_id in results.results_test:
    d_train = results.results_train[shift_id, sample_id, run_id]
    d_test = results.results_test[shift_id, sample_id, run_id]
    d_val = results.results_val[shift_id, sample_id, run_id]

    results_array[sample_id, shift_id, run_id, 0] = d_test.auc - d_val.auc
    results_array[sample_id, shift_id, run_id, 1] = d_test.smr - d_val.smr
    results_array[sample_id, shift_id, run_id, 2] = d_test.eo
    results_array[sample_id, shift_id, run_id, 3] = d_test.dp
    results_array[sample_id, shift_id, run_id, 4] = d_test.brier - d_val.brier
    results_array[sample_id, shift_id, run_id, 5] = d_test.citl
    results_array[sample_id, shift_id, run_id, 6] = d_test.cs
    results_array[sample_id, shift_id, run_id, 7] = d_test.fnr
    results_array[sample_id, shift_id, run_id, 8] = d_test.citl_diff
    results_array[sample_id, shift_id, run_id, 9] = d_test.cs_diff

  return results_array, metric_names


def get_brier_score(target, probabilities):
  return brier_score_loss(target, probabilities)


'''Expected calibration error described in https://arxiv.org/abs/1906.02530
Taken from https://github.com/google-research/google-research/blob/master/uq_benchmark_2019/metrics_lib.py
'''
def bin_predictions_and_accuracies(probabilities, ground_truth, bins=10):
  """A helper function which histograms a vector of probabilities into bins.
  Args:
    probabilities: A numpy vector of N probabilities assigned to each prediction
    ground_truth: A numpy vector of N ground truth labels in {0,1}
    bins: Number of equal width bins to bin predictions into in [0, 1], or an
      array representing bin edges.
  Returns:
    bin_edges: Numpy vector of floats containing the edges of the bins
      (including leftmost and rightmost).
    accuracies: Numpy vector of floats for the average accuracy of the
      predictions in each bin.
    counts: Numpy vector of ints containing the number of examples per bin.
  """
  _validate_probabilities(probabilities)
  _check_rank_nonempty(rank=1,
                       probabilities=probabilities,
                       ground_truth=ground_truth)

  if len(probabilities) != len(ground_truth):
    raise ValueError(
        'Probabilies and ground truth must have the same number of elements.')

  if [v for v in ground_truth if v not in [0., 1., True, False]]:
    raise ValueError(
        'Ground truth must contain binary labels {0,1} or {False, True}.')

  if isinstance(bins, int):
    num_bins = bins
  else:
    num_bins = bins.size - 1

  # Ensure probabilities are never 0, since the bins in np.digitize are open on
  # one side.
  probabilities = np.where(probabilities == 0, 1e-8, probabilities)
  counts, bin_edges = np.histogram(probabilities, bins=bins, range=[0., 1.])
  indices = np.digitize(probabilities, bin_edges, right=True)
  accuracies = np.array([np.mean(ground_truth[indices == i])
                         for i in range(1, num_bins + 1)])
  return bin_edges, accuracies, counts

'''Expected calibration error described in https://arxiv.org/abs/1906.02530
Taken from https://github.com/google-research/google-research/blob/master/uq_benchmark_2019/metrics_lib.py
'''
def bin_centers_of_mass(probabilities, bin_edges):
    probabilities = np.where(probabilities == 0, 1e-8, probabilities)
    indices = np.digitize(probabilities, bin_edges, right=True)
    return np.array([np.mean(probabilities[indices == i])
                    for i in range(1, len(bin_edges))])

'''Expected calibration error described in https://arxiv.org/abs/1906.02530
Taken from https://github.com/google-research/google-research/blob/master/uq_benchmark_2019/metrics_lib.py
'''
def expected_calibration_error(ground_truth, probabilities, bins=15):
    """Compute the expected calibration error of a set of preditions in [0, 1].
    Args:
    probabilities: A numpy vector of N probabilities assigned to each prediction
    ground_truth: A numpy vector of N ground truth labels in {0,1, True, False}
    bins: Number of equal width bins to bin predictions into in [0, 1], or
        an array representing bin edges.
    Returns:
    Float: the expected calibration error.
    """

    probabilities = probabilities.flatten()
    ground_truth = ground_truth.flatten()
    bin_edges, accuracies, counts = bin_predictions_and_accuracies(
        probabilities, ground_truth, bins)
    bin_centers = bin_centers_of_mass(probabilities, bin_edges)
    num_examples = np.sum(counts)

    ece = np.sum([(counts[i] / float(num_examples)) * np.sum(
        np.abs(bin_centers[i] - accuracies[i]))
                for i in range(bin_centers.size) if counts[i] > 0])
    return ece


def logit(p):
  return np.log(p/(1-p))


def odds_ratio(p):
  return p/(1-p)


def calibration_in_the_large(ground_truth, probabilities):
  or_probabilities = odds_ratio(np.mean(probabilities))
  or_ground_truth = odds_ratio(np.mean(ground_truth))

  return or_probabilities/or_ground_truth


def calibration_slope(ground_truth, probabilities):
  probabilities = np.array(probabilities)
  logit_probabilities = logit(probabilities).reshape(-1,1)
  lr = LogisticRegression(penalty='none', fit_intercept=True).fit(logit_probabilities, ground_truth)

  return lr.coef_.item()


# def max_equalized_odds_violation(target, predictions, sensitive_feature):
#     '''
#     Maximum violation of equalized odds constraint. From fair reductions paper,
#     max_{y,a} |E[h(X)|Y=y,A=a]-E[h(X)|Y=y]|
#     :param sensitive_feature: actual value of the sensitive feature
#     '''
#     tpr = group_recall_score(target, predictions, sensitive_feature)
#     specificity = group_specificity_score(target, predictions, sensitive_feature) # 1-fpr
    
#     max_violation = max([abs(tpr_group-tpr.overall) for tpr_group in tpr.by_group.values()] +
#         [abs(spec_group-specificity.overall) for spec_group in specificity.by_group.values()])
    
#     return max_violation


# def max_demography_parity_violation(target, predictions, sensitive_feature):
#     '''
#     Maximum violation of demographic parity constraint.
#     max_{a} |E[h(X)|A=a]-min_{a} |E[h(X)|A=a]
#     :param sensitive_feature: actual value of the sensitive feature
#     '''
#     acc = group_mean_prediction(target, predictions, sensitive_feature)
#     acc_ad = [i for i in acc.by_group.values()]
#     max_violation = abs(acc_ad[0]-acc_ad[1])
    
#     return max_violation


def get_equalized_odds_difference(target, predictions, sensitive_feature):
  return equalized_odds_difference(target, predictions, sensitive_features=sensitive_feature)


def get_demography_parity_difference(target, predictions, sensitive_feature):
  return demographic_parity_difference(target, predictions, sensitive_features=sensitive_feature)


def get_false_negative_rate_difference(target, predictions, sensitive_feature):
  metric_fns = {'false_negative_rate': false_negative_rate}
  group = MetricFrame(metric_fns,
                      target, predictions,
                      sensitive_features=sensitive_feature)
  return group.difference(method='between_groups').item()


def get_calibration_in_the_large_difference(ground_truth, probabilities, sensitive_feature):
  metric_fns = {'calibration_in_the_large': calibration_in_the_large}
  group = MetricFrame(metric_fns,
                      ground_truth, probabilities,
                      sensitive_features=sensitive_feature)
  return group.difference(method='between_groups').item()


def get_calibration_slope_difference(ground_truth, probabilities, sensitive_feature):
  metric_fns = {'calibration_slope': calibration_slope}
  group = MetricFrame(metric_fns,
                      ground_truth, probabilities,
                      sensitive_features=sensitive_feature)
  return group.difference(method='between_groups').item()


def _validate_probabilities(probabilities, multiclass=False):
  if np.max(probabilities) > 1. or np.min(probabilities) < 0.:
    raise ValueError('All probabilities must be in [0,1].')
  if multiclass and not np.allclose(1, np.sum(probabilities, axis=-1),
                                    atol=1e-5):
    raise ValueError(
        'Multiclass probabilities must sum to 1 along the last dimension.')


def _check_rank_nonempty(rank, **kwargs):
  for key, array in six.iteritems(kwargs):
    if len(array) <= 1 or array.ndim != rank:
      raise ValueError(
          '%s must be a rank-1 array of length > 1; actual shape is %s.' %
          (key, array.shape))


def make_keras_picklable():
    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            keras.models.save_model(self, fd.name, overwrite=True)
            model_str = fd.read()
        d = { 'model_str': model_str }
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()
            model = keras.models.load_model(fd.name)
        self.__dict__ = model.__dict__


    cls = keras.models.Model
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__