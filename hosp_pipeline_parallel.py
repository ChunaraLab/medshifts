'''
Modifed from code by Stephan Rabanser https://github.com/steverab/failing-loudly

Detect shifts, train and test models across hospitals and regions
Runs parallely across 36 processes

Usage:

for multivaritate tests:
# region
python hosp_pipeline_parallel.py --datset eicu --path orig --test_type multiv --missing_imp mean --num_hosp 4 --random_runs 100 --min_samples 5000 --sens_attr race --group --group_type regions --limit_samples
# beds
python hosp_pipeline_parallel.py --datset eicu --path orig --test_type multiv --missing_imp mean --num_hosp 4 --random_runs 100 --min_samples 10000 --sens_attr race --group --group_type beds --limit_samples
# region, beds
python hosp_pipeline_parallel.py --datset eicu --path orig --test_type multiv --missing_imp mean --num_hosp 5 --random_runs 100 --min_samples 5000 --sens_attr race --group --group_type regions_beds --limit_samples
# region, beds, teaching
python hosp_pipeline_parallel.py --datset eicu --path orig --test_type multiv --missing_imp mean --num_hosp 6 --random_runs 100 --min_samples 4000 --sens_attr race --group --group_type regions_beds_teaching --limit_samples
# hospitals
python hosp_pipeline_parallel.py --datset eicu --path orig --test_type multiv --missing_imp mean --num_hosp 10 --random_runs 100 --min_samples 1631 --sens_attr race --limit_samples
# python hosp_pipeline_parallel.py --datset eicu --path orig --test_type multiv --missing_imp mean --num_hosp 10 --random_runs 100 --min_samples 2000 --sens_attr race --limit_samples

for univaritate tests:
python hosp_pipeline_parallel.py eicu orig univ mean
'''

import argparse
import pickle
import numpy as np
from itertools import combinations
import time
seed = 1
np.random.seed(seed)

from shift_detector import *
from shift_locator import *
from shift_applicator import *
from data_utils import *
import os
import sys
from exp_utils import *
from plot_utils import *

import multiprocessing
from joblib import Parallel, delayed
num_cores = min(36, multiprocessing.cpu_count())

# -------------------------------------------------
# CONFIG
# -------------------------------------------------

# make_keras_picklable()
np.set_printoptions(threshold=sys.maxsize)

parser = argparse.ArgumentParser()
parser.add_argument("--datset", type=str, default='eicu')
parser.add_argument("--path", type=str, default='orig')
parser.add_argument("--test_type", type=str, default='multiv')
parser.add_argument("--missing_imp", type=str, default='mean')
parser.add_argument("--sens_attr", type=str, default='gender') # gender, race
parser.add_argument("--num_hosp", type=int, default=5)
parser.add_argument("--random_runs", type=int, default=10)
parser.add_argument("--min_samples", type=int, default=5000)
parser.add_argument("--group", action='store_true')
parser.add_argument("--group_type", type=str, default='hosp')
parser.add_argument("--limit_samples", action='store_true') # limit two-sample testing to 5000 samples
args = parser.parse_args()
# args = parser.parse_args("--datset eicu --path orig --test_type multiv\
#                         --missing_imp mean --num_hosp 2 --random_runs 1\
#                         --min_samples 10000 --sens_attr race --group".split())
print("Arguments", args)

datset = args.datset
test_type = args.test_type
missing_imp = args.missing_imp
use_group = args.group
group_type = args.group_type
sens_attr = args.sens_attr
limit_samples = args.limit_samples

HospitalGroups_eicu, HospitalGroupsColnames_eicu = get_groups_colnames(group_type)

outdir = '/data/hs3673/medshifts/failing-loudly/'
# path = './hosp_results_gossis_multiv/'
path = outdir+'hosp_results_{}_{}_shuffle/'.format(datset, test_type)
path += '{}_group{}_{}_nh{}_run{}_mins{}_s{}_l{}_{}/'.format(datset, use_group, group_type, args.num_hosp, args.random_runs, args.min_samples, sens_attr, limit_samples, args.path)
print("path", path)

if not os.path.exists(path):
    os.makedirs(path)

# Define train-test pairs of hospitals
NUM_HOSPITALS_TOP = args.num_hosp #2 # hospitals with records >= 1000
hosp_pairs = []
# TODO move to data_utils
if datset =='eicu':
    if use_group:
        HospitalIDs = HospitalGroups_eicu
    else: # single hospitals
        HospitalIDs = HospitalIDs_eicu
    FeatureGroups = FeatureGroups_eicu

    # Define feature groups
    # feature_groups = [['labs','vitals','demo','others','saps2diff']]
    # feature_groups = [['labs','labs_blood_gas']]
    # feature_groups = [['vitals']]
    # feature_groups = [['demo']]
    # feature_groups = [['saps2labs','saps2vitals']]
    # feature_groups = [['labs'], ['vitals'], ['demo']]
    # feature_groups = [['saps2'], ['labs'], ['vitals'], ['demo']]
    feature_groups = [['saps2']]
    # feature_groups = [['saps2'], ['labs','vitals','demo','others']]
elif datset =='gossis':
    HospitalIDs = HospitalIDs_gossis
    FeatureGroups = FeatureGroups_gossis
    
    # Define feature groups
    feature_groups = [['APACHE_covariate']]
    # feature_groups = [['demographic'], ['vitals'], ['labs','labs_blood_gas'],['APACHE_covariate']]
    # feature_groups = [['APACHE_covariate'], ['labs','labs_blood_gas'], ['vitals'], ['APACHE_comorbidity'],
    #                     ['demographic','vitals','labs','labs_blood_gas','APACHE_comorbidity']]

HospitalIDs = HospitalIDs[:NUM_HOSPITALS_TOP]
for hi in HospitalIDs:
    for hj in HospitalIDs:
        hosp_pairs.append(([hi],[hj]))
# hosp_pairs = [([394],[416])]
print('Use groups', use_group, 'Sensitive attribute', sens_attr, 'Hospital pairs', hosp_pairs)

# Define DR methods
# dr_techniques = [DimensionalityReduction.NoRed.value, DimensionalityReduction.PCA.value, DimensionalityReduction.SRP.value, DimensionalityReduction.UAE.value, DimensionalityReduction.TAE.value, DimensionalityReduction.BBSDs.value, DimensionalityReduction.BBSDh.value]
dr_techniques = [DimensionalityReduction.NoRed.value]
# dr_techniques = [DimensionalityReduction.NoRed.value, DimensionalityReduction.PCA.value]
# dr_techniques = [DimensionalityReduction.NoRed.value, DimensionalityReduction.PCA.value, DimensionalityReduction.SRP.value]
if test_type == 'multiv':
    # dr_techniques = [DimensionalityReduction.NoRed.value, DimensionalityReduction.PCA.value, DimensionalityReduction.SRP.value, DimensionalityReduction.UAE.value, DimensionalityReduction.TAE.value, DimensionalityReduction.BBSDs.value]
    dr_techniques = [DimensionalityReduction.NoRed.value]
    # dr_techniques = [DimensionalityReduction.NoRed.value, DimensionalityReduction.PCA.value]
    # dr_techniques = [DimensionalityReduction.NoRed.value, DimensionalityReduction.PCA.value, DimensionalityReduction.SRP.value]
if test_type == 'univ':
    dr_techniques_plot = dr_techniques.copy()
    # dr_techniques_plot.append(DimensionalityReduction.Classif.value)
else:
    dr_techniques_plot = dr_techniques.copy()

# Define test types and general test sample sizes
test_types = [td.value for td in TestDimensionality]
if test_type == 'multiv':
    od_tests = []
    md_tests = [MultidimensionalTest.MMD.value]
    samples = [args.min_samples]
    # samples = [2500]
    # samples = [-1]
    # samples = [1000, 1500, -1]
else:
    # od_tests = [od.value for od in OnedimensionalTest]
    od_tests = [OnedimensionalTest.KS.value]
    md_tests = []
    samples = [args.min_samples]
    # samples = [2500]
    # samples = [-1]
    # samples = [1000, 1500, -1]
difference_samples = 10

if missing_imp == 'mice':
    missing_techniques = ['mean', 'mice']
else:
    missing_techniques = ['mean']

# Number of random runs to average results over    
random_runs = args.random_runs # 5

# Signifiance level
# sign_level = 0.05
sign_level = 0.01

# Define shift types
# if args.path == 'orig': # sys.argv[2]
#     shifts = ['orig']
#     brightness = [0.75]
#     # shifts = ['rand', 'orig']
#     # brightness = [1.25, 0.75]
# else:
#     shifts = []
shifts = ['orig']

# -------------------------------------------------
# PIPELINE START
# -------------------------------------------------

def test_hosp_pair(df, target, features, feature_group, hosp_train, hosp_test, use_group, sens_attr):
    print("\n========\nFeature Set, Hosp Train, Hosp Test", target, feature_group, hosp_train, hosp_test)
    print("========\n")

    hosp_folder_name = 'tr_' + '_'.join(map(str, hosp_train)) + '_ts_' + '_'.join(map(str, hosp_test))
    hosp_path = path + "_".join(feature_group) + '/' + hosp_folder_name + '/'
    if not os.path.exists(hosp_path):
        os.makedirs(hosp_path)

    metric_results = MetricResults(len(shifts), len(samples), random_runs)

    samples_shifts_rands_dr_tech = np.ones((len(samples), len(shifts), random_runs, len(dr_techniques_plot))) * (-1) # TODO add hosp_pair
    samples_shifts_rands_dr_tech_t_val = np.ones((len(samples), len(shifts), random_runs, len(dr_techniques_plot))) * (-1) # TODO add hosp_pair
    
    samples_shifts_rands_feat_p_vals = np.ones((len(samples), len(shifts), len(dr_techniques_plot), len(od_tests), len(features), random_runs)) * (-1)
    samples_shifts_rands_feat_t_vals = np.ones((len(samples), len(shifts), len(dr_techniques_plot), len(od_tests), len(features), random_runs)) * (-1)

    for shift_idx, shift in enumerate(shifts):

        shift_path =  hosp_path + shift + '/'
        if not os.path.exists(shift_path):
            os.makedirs(shift_path)

        rand_run_p_vals = np.ones((len(samples), len(dr_techniques_plot), random_runs)) * (-1)
        rand_run_t_vals = np.ones((len(samples), len(dr_techniques_plot), random_runs)) * (-1)

        rand_run_feat_p_vals = np.ones((len(samples), len(dr_techniques_plot), len(od_tests), len(features), random_runs)) * (-1)
        rand_run_feat_t_vals = np.ones((len(samples), len(dr_techniques_plot), len(od_tests), len(features), random_runs)) * (-1)

        for rand_run in range(random_runs):

            print("\nRandom run %s" % rand_run)

            rand_run_path = shift_path + str(rand_run) + '/'
            if not os.path.exists(rand_run_path):
                os.makedirs(rand_run_path)

            np.random.seed(rand_run)

            # Load data
            time_now = time.time()
            # print('Original')
            (X_tr_orig, y_tr_orig, sens_tr_orig), (X_val_orig, y_val_orig, sens_val_orig), (X_te_orig, y_te_orig, sens_te_orig), orig_dims, nb_classes = load_hosp_dataset(datset, df, target, features, hosp_train, hosp_test, use_group, sens_attr, shuffle=False)

            X_te_1 = X_te_orig.copy()
            y_te_1 = y_te_orig.copy()
            sens_te_1 = sens_te_orig.copy()

            # Apply shift
            if shift != 'orig':
                (X_tr_orig, y_tr_orig, sens_tr_orig), (X_val_orig, y_val_orig, sens_val_orig), (X_te_orig, y_te_orig, sens_te_orig), orig_dims, nb_classes = load_hosp_dataset(datset, df, target, features, hosp_train, hosp_test, use_group, sens_attr, shuffle=True)

                (X_te_1, y_te_1) = apply_shift(X_te_orig, y_te_orig, shift, orig_dims, datset)

            X_te_2 , y_te_2, sens_te_2 = random_shuffle(X_te_1, y_te_1, sens_te_1)

            red_dim = -1
            red_models = [None] * len(DimensionalityReduction) # new model for each shift, random run
            
            time_diff = time.time() - time_now
            print('Time: {}s taken in reading data\n{}, {}, {}'.format(time_diff, rand_run, hosp_train, hosp_test))

            # Check detection performance for different numbers of samples from test
            for si, sample in enumerate(samples):

                # print("Sample %s" % sample)

                sample_path = rand_run_path + str(sample) + '/'
                if not os.path.exists(sample_path):
                    os.makedirs(sample_path)

                # Reduce number of test samples
                if sample==-1: # use all test and train
                    X_te_3 = X_te_2
                    y_te_3 = y_te_2
                    sens_te_3 = sens_te_2

                    X_val_3 = X_val_orig
                    y_val_3 = y_val_orig
                    sens_val_3 = sens_val_orig

                    X_tr_3 = np.copy(X_tr_orig)
                    y_tr_3 = np.copy(y_tr_orig)
                    sens_tr_3 = np.copy(sens_tr_orig)
                else: # reduce test and train to same number of samples
                    X_te_3 = X_te_2[:sample,:]
                    y_te_3 = y_te_2[:sample]
                    sens_te_3 = sens_te_2[:sample]
                
                    X_val_3 = X_val_orig[:sample,:]
                    y_val_3 = y_val_orig[:sample]
                    sens_val_3 = sens_val_orig[:sample]

                    X_tr_3 = np.copy(X_tr_orig[:sample,:])
                    y_tr_3 = np.copy(y_tr_orig[:sample])
                    sens_tr_3 = np.copy(sens_tr_orig[:sample])
                
                # X_tr_3 = np.copy(X_tr_orig)
                # y_tr_3 = np.copy(y_tr_orig)

                # Detect shift
                time_now = time.time()
                shift_detector = ShiftDetector(dr_techniques, test_types, od_tests, md_tests, sign_level, red_models, sample, datset, limit_samples)
                (od_decs, ind_od_decs, ind_od_p_vals, ind_od_t_vals, ind_od_feat_p_vals, ind_od_feat_t_vals),\
                (md_decs, ind_md_decs, ind_md_p_vals, ind_md_t_vals),\
                red_dim, red_models, d_tr, d_te, d_val\
                 = shift_detector.detect_data_shift(X_tr_3, y_tr_3, sens_tr_3, X_val_3, y_val_3, sens_val_3, X_te_3, y_te_3, sens_te_3, orig_dims, nb_classes)

                time_diff = time.time() - time_now
                print('Time: {}s taken in detecting shift\n{}, {}, {}'.format(time_diff, rand_run, hosp_train, hosp_test))

                metric_results.add(d_tr, d_te, d_val, shift_idx, si, rand_run)

                if test_type == 'multiv':
                    # print("Shift decision: ", ind_md_decs.flatten())
                    # print("Shift p-vals: ", ind_md_p_vals.flatten())

                    rand_run_p_vals[si,:,rand_run] = ind_md_p_vals.flatten()
                    rand_run_t_vals[si,:,rand_run] = ind_md_t_vals.flatten()
                else:
                    # print("Shift decision: ", ind_od_decs.flatten())
                    # print("Shift p-vals: ", ind_od_p_vals.flatten())
                    
                    if DimensionalityReduction.Classif.value not in dr_techniques_plot:
                        rand_run_p_vals[si,:,rand_run] = ind_od_p_vals.flatten()

                        rand_run_feat_p_vals[si, :, :, :, rand_run] = ind_od_feat_p_vals
                        rand_run_feat_t_vals[si, :, :, :, rand_run] = ind_od_feat_t_vals
                        continue

                    # Characterize shift via domain classifier
                    # shift_locator = ShiftLocator(orig_dims, dc=DifferenceClassifier.FFNNDCL, sign_level=sign_level)
                    shift_locator = ShiftLocator(orig_dims, dc=DifferenceClassifier.FLDA, sign_level=sign_level)
                    model, score, (X_tr_dcl, y_tr_dcl, y_tr_old, X_te_dcl, y_te_dcl, y_te_old) = shift_locator.build_model(X_tr_3, y_tr_3, X_te_3, y_te_3)
                    test_indices, test_perc, dec, p_val = shift_locator.most_likely_shifted_samples(model, X_te_dcl, y_te_dcl)

                    rand_run_p_vals[si,:,rand_run] = np.append(ind_od_p_vals.flatten(), p_val)
                    rand_run_t_vals[si,:,rand_run] = np.append(ind_od_t_vals.flatten(), -1) # TODO change t_val for Classif dr method

            np.savetxt("%s/dr_method_p_vals.csv" % rand_run_path, rand_run_p_vals[:,:,rand_run], delimiter=",")
            np.savetxt("%s/dr_method_t_vals.csv" % rand_run_path, rand_run_t_vals[:,:,rand_run], delimiter=",")

            np.save("%s/dr_method_feat_p_vals.npy" % rand_run_path, rand_run_feat_p_vals[:,:,:,:,rand_run])
            np.save("%s/dr_method_feat_t_vals.npy" % rand_run_path, rand_run_feat_t_vals[:,:,:,:,rand_run])

        for dr_idx, dr in enumerate(dr_techniques_plot):
            samples_shifts_rands_dr_tech[:,shift_idx,:,dr_idx] = rand_run_p_vals[:,dr_idx,:]
            samples_shifts_rands_dr_tech_t_val[:,shift_idx,:,dr_idx] = rand_run_t_vals[:,dr_idx,:]

            samples_shifts_rands_feat_p_vals[:,shift_idx,dr_idx,:,:,:] = rand_run_feat_p_vals[:,dr_idx,:,:,:]
            samples_shifts_rands_feat_t_vals[:,shift_idx,dr_idx,:,:,:] = rand_run_feat_t_vals[:,dr_idx,:,:,:]

        np.save("%s/samples_shifts_rands_dr_tech.npy" % (hosp_path), samples_shifts_rands_dr_tech)
        np.save("%s/samples_shifts_rands_dr_tech_t_val.npy" % (hosp_path), samples_shifts_rands_dr_tech_t_val)

        with open("%s/samples_shifts_rands_metrics.pkl" % (hosp_path), 'wb') as fw:
            pickle.dump(metric_results, fw)

        np.save("%s/samples_shifts_rands_feat_p_vals.npy" % (hosp_path), samples_shifts_rands_feat_p_vals)
        np.save("%s/samples_shifts_rands_feat_t_vals.npy" % (hosp_path), samples_shifts_rands_feat_t_vals)

    np.save("%s/samples_shifts_rands_dr_tech.npy" % (hosp_path), samples_shifts_rands_dr_tech)
    np.save("%s/samples_shifts_rands_dr_tech_t_val.npy" % (hosp_path), samples_shifts_rands_dr_tech_t_val)
    
    with open("%s/samples_shifts_rands_metrics.pkl" % (hosp_path), 'wb') as fw:
        pickle.dump(metric_results, fw)

    np.save("%s/samples_shifts_rands_feat_p_vals.npy" % (hosp_path), samples_shifts_rands_feat_p_vals)
    np.save("%s/samples_shifts_rands_feat_t_vals.npy" % (hosp_path), samples_shifts_rands_feat_t_vals)

if __name__ == "__main__":

    df = import_hosp_dataset(datset)

    for feature_set_idx, feature_group in enumerate(feature_groups):

        target = FeatureGroups['outcome']
        feature_set = []
        for group in feature_group:
            # All features in group
            feature_set += FeatureGroups[group]

            # # Univariate
            # for feats in FeatureGroups[group]:
            #     feature_sets.append([feats])

            # # Pairs of features in group
            # for subs in combinations(FeatureGroups[group], 2):
            #     feature_sets.append(list(subs)) # TODO de-duplicate

        Parallel(n_jobs=num_cores)(delayed(test_hosp_pair)(df, target, feature_set, feature_group,\
                                            hosp_train, hosp_test, use_group, sens_attr) for (hosp_train, hosp_test) in hosp_pairs)