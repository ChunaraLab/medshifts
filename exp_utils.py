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
from sklearn.metrics import brier_score_loss, roc_auc_score

import types
import tempfile
import keras.models


Metrics = namedtuple('Metrics',('brier', 'mae', 'logloss',
                                'auc', 'acc', 'smr', 'citl', 'cs', 'ece',
                                'eo', 'dp', 'fnr', 'citl_diff', 'cs_diff',
                                'fnr_min', 'fnr_maj',
                                'cs_min', 'cs_maj',
                                'citl_min', 'citl_maj',
                                'auc_diff', 'auc_min', 'auc_maj',
                                'prob','pred','y','sens'))


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
                citl_diff=d_train['citl_diff'], cs_diff=d_train['cs_diff'],
                fnr_min=d_train['fnr_min'], fnr_maj=d_train['fnr_maj'],
                cs_min=d_train['cs_min'], cs_maj=d_train['cs_maj'],
                citl_min=d_train['citl_min'], citl_maj=d_train['citl_maj'],
                auc_diff=d_train['auc_diff'], auc_min=d_train['auc_min'], auc_maj=d_train['auc_maj'],
                prob=d_train['prob'], pred=d_train['pred'], y=d_train['y'], sens=d_train['sens'])

    d_test_ = Metrics(brier=d_test['brier'], mae=d_test['mae'], logloss=d_test['logloss'],
                auc=d_test['auc'], acc=d_test['acc'],
                smr=d_test['smr'], citl=d_test['citl'], cs=d_test['cs'], ece=d_test['ece'],
                eo=d_test['eo'], dp=d_test['dp'], fnr=d_test['fnr'],
                citl_diff=d_test['citl_diff'], cs_diff=d_test['cs_diff'],
                fnr_min=d_test['fnr_min'], fnr_maj=d_test['fnr_maj'],
                cs_min=d_test['cs_min'], cs_maj=d_test['cs_maj'],
                citl_min=d_test['citl_min'], citl_maj=d_test['citl_maj'],
                auc_diff=d_test['auc_diff'], auc_min=d_test['auc_min'], auc_maj=d_test['auc_maj'],
                prob=d_test['prob'], pred=d_test['pred'], y=d_test['y'], sens=d_test['sens'])

    d_val_ = Metrics(brier=d_val['brier'], mae=d_val['mae'], logloss=d_val['logloss'],
                auc=d_val['auc'], acc=d_val['acc'],
                smr=d_val['smr'], citl=d_val['citl'], cs=d_val['cs'], ece=d_val['ece'],
                eo=d_val['eo'], dp=d_val['dp'], fnr=d_val['fnr'],
                citl_diff=d_val['citl_diff'], cs_diff=d_val['cs_diff'],
                fnr_min=d_val['fnr_min'], fnr_maj=d_val['fnr_maj'],
                cs_min=d_val['cs_min'], cs_maj=d_val['cs_maj'],
                citl_min=d_val['citl_min'], citl_maj=d_val['citl_maj'],
                auc_diff=d_val['auc_diff'], auc_min=d_val['auc_min'], auc_maj=d_val['auc_maj'],
                prob=d_val['prob'], pred=d_val['pred'], y=d_val['y'], sens=d_val['sens'])

    self.results_train[shift_id, sample_id, run_id] = d_train_
    self.results_test[shift_id, sample_id, run_id] = d_test_
    self.results_val[shift_id, sample_id, run_id] = d_val_


def get_metrics_array(results):
  num_shifts = results.num_shifts
  num_sample_choices = results.num_sample_choices
  num_runs = results.num_runs
  metric_names = ['aucdiff','smrdiff','eo','dp','brierdiff',
                  'citl','cs','fnr','citldisp','csdisp',
                  'auc', 'smr', 'brier', 'acc', 'csdiff',
                  'aucval', 'csval', 'fnrval', 'csdispval',
                  'auctrain', 'cstrain', 'fnrtrain', 'csdisptrain',
                  'fnrmin', 'fnrmaj', 'csmin', 'csmaj', 'citlmin', 'citlmaj',
                  'aucdisp', 'aucmin', 'aucmaj',
                  'fnrsign','csdispsign','citldispsign', 'aucdispsign']
  results_array = np.ones((num_sample_choices, num_shifts, num_runs, len(metric_names))) * (-1) # 0-auc, 1-smr, 2-eo, 3-dp
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
    results_array[sample_id, shift_id, run_id, 10] = d_test.auc
    results_array[sample_id, shift_id, run_id, 11] = d_test.smr
    results_array[sample_id, shift_id, run_id, 12] = d_test.brier
    results_array[sample_id, shift_id, run_id, 13] = d_test.acc
    results_array[sample_id, shift_id, run_id, 14] = d_test.cs - d_val.cs
    results_array[sample_id, shift_id, run_id, 15] = d_val.auc
    results_array[sample_id, shift_id, run_id, 16] = d_val.cs
    results_array[sample_id, shift_id, run_id, 17] = d_val.fnr
    results_array[sample_id, shift_id, run_id, 18] = d_val.cs_diff
    results_array[sample_id, shift_id, run_id, 19] = d_train.auc
    results_array[sample_id, shift_id, run_id, 20] = d_train.cs
    results_array[sample_id, shift_id, run_id, 21] = d_train.fnr
    results_array[sample_id, shift_id, run_id, 22] = d_train.cs_diff
    results_array[sample_id, shift_id, run_id, 23] = d_test.fnr_min
    results_array[sample_id, shift_id, run_id, 24] = d_test.fnr_maj
    results_array[sample_id, shift_id, run_id, 25] = d_test.cs_min
    results_array[sample_id, shift_id, run_id, 26] = d_test.cs_maj
    results_array[sample_id, shift_id, run_id, 27] = d_test.citl_min
    results_array[sample_id, shift_id, run_id, 28] = d_test.citl_maj
    results_array[sample_id, shift_id, run_id, 29] = d_test.auc_diff
    results_array[sample_id, shift_id, run_id, 30] = d_test.auc_min
    results_array[sample_id, shift_id, run_id, 31] = d_test.auc_maj
    results_array[sample_id, shift_id, run_id, 32] = d_test.fnr_min - d_test.fnr_maj
    results_array[sample_id, shift_id, run_id, 33] = d_test.cs_min - d_test.cs_maj
    results_array[sample_id, shift_id, run_id, 34] = d_test.citl_min - d_test.citl_maj
    results_array[sample_id, shift_id, run_id, 35] = d_test.auc_min - d_test.auc_maj

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


def get_roc_auc_score_difference(target, probabilities, sensitive_feature):
  metric_fns = {'roc_auc_score': roc_auc_score}
  group = MetricFrame(metric_fns,
                      target, probabilities,
                      sensitive_features=sensitive_feature)
  auc_min = group.by_group.loc[0].item()
  auc_maj = group.by_group.loc[1].item()
  return group.difference(method='between_groups').item(), auc_min, auc_maj


def get_false_negative_rate_difference(target, predictions, sensitive_feature):
  metric_fns = {'false_negative_rate': false_negative_rate}
  group = MetricFrame(metric_fns,
                      target, predictions,
                      sensitive_features=sensitive_feature)
  fnr_min = group.by_group.loc[0].item()
  fnr_maj = group.by_group.loc[1].item()
  return group.difference(method='between_groups').item(), fnr_min, fnr_maj


def get_calibration_in_the_large_difference(ground_truth, probabilities, sensitive_feature):
  metric_fns = {'calibration_in_the_large': calibration_in_the_large}
  group = MetricFrame(metric_fns,
                      ground_truth, probabilities,
                      sensitive_features=sensitive_feature)
  citl_min = group.by_group.loc[0].item()
  citl_maj = group.by_group.loc[1].item()
  return group.difference(method='between_groups').item(), citl_min, citl_maj


def get_calibration_slope_difference(ground_truth, probabilities, sensitive_feature):
  metric_fns = {'calibration_slope': calibration_slope}
  group = MetricFrame(metric_fns,
                      ground_truth, probabilities,
                      sensitive_features=sensitive_feature)
  cs_min = group.by_group.loc[0].item()
  cs_maj = group.by_group.loc[1].item()
  return group.difference(method='between_groups').item(), cs_min, cs_maj


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