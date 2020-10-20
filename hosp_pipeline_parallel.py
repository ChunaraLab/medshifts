'''
Written by Stephan Rabanser https://github.com/steverab/failing-loudly
Modifed

Detect shifts across hospitals

Usage
for multivaritate tests:
python hosp_pipeline_parallel.py --datset eicu --path orig --test_type multiv --missing_imp mean --num_hosp 5 --random_runs 10 --min_samples 5000 --group

for univaritate tests:
python hosp_pipeline_parallel.py eicu orig univ mean

# TODO
quantify shift by discriminate between train test
sample size vs acc, smr Johnson plot with all train in X_tr_3, y_tr_3
frequency univariate
colorcode scatterplot by hospital meta data
mice
interpretable distribution change code
mean_p_vals = -1 for 73, 338
reduce dimension of X_te using model trained on X_te in shift_detector
acc for dimension reduction also in shift_detector
apache 4 feature group in apacheapsvar, apachepredvar SQL tables

get HospitalIDs and FeatureGroups from function in data_utils
record train set accuracy
use validation set for accuracy in shift_detector
DimensionalityReduction.NoRed in shift_reductor used to calculate accuracy. calculate in separate class. reduce return prob instead of pred
impute missing values in shift_reductor pca, srp, lda
number of dims in shift_detector
shift_tester.test_shift one dim check if t_val correct after FWER correction
shift_tester.test_chi2_shift one dim return t_val
use validation set
load data once
move filename in data_utils to main file
2 rows less, 70128 orig vs now 70126
'''

import argparse
import numpy as np
import tensorflow
from itertools import combinations
import time
seed = 1
np.random.seed(seed)
tensorflow.random.set_seed(seed)

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
num_cores = min(25, multiprocessing.cpu_count())

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
parser.add_argument("--num_hosp", type=int, default=5)
parser.add_argument("--random_runs", type=int, default=10)
parser.add_argument("--min_samples", type=int, default=5000)
parser.add_argument("--group", action='store_true')
args = parser.parse_args()

datset = args.datset
test_type = args.test_type
missing_imp = args.missing_imp
use_group = args.group

# path = './hosp_results_gossis_multiv/'
path = './hosp_results_{}_{}/'.format(datset, test_type)
path += '{}_nh{}_run{}_mins{}_{}/'.format(datset, args.num_hosp, args.random_runs, args.min_samples, args.path)

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
    # feature_groups = [['demographic']]
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
print('Use groups', use_group, 'Hospital pairs', hosp_pairs)

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
sign_level = 0.05

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

def test_hosp_pair(df, target, features, feature_group, hosp_train, hosp_test, use_group):
    print("\n========\nFeature Set, Hosp Train, Hosp Test", target, feature_group, hosp_train, hosp_test)
    print("========\n")

    hosp_folder_name = 'tr_' + '_'.join(map(str, hosp_train)) + '_ts_' + '_'.join(map(str, hosp_test))
    hosp_path = path + "_".join(feature_group) + '/' + hosp_folder_name + '/'
    if not os.path.exists(hosp_path):
        os.makedirs(hosp_path)

    samples_shifts_rands_dr_tech = np.ones((len(samples), len(shifts), random_runs, len(dr_techniques_plot))) * (-1) # TODO add hosp_pair
    samples_shifts_rands_dr_tech_t_val = np.ones((len(samples), len(shifts), random_runs, len(dr_techniques_plot))) * (-1) # TODO add hosp_pair
    samples_shifts_rands_te_acc = np.ones((len(samples), len(shifts), random_runs, 4)) * (-1) # 0-auc, 1-smr, 2-eo, 3-dp # TODO add auc, smr, p-val, mmd in same array. add hosp_pair

    samples_shifts_rands_feat_p_vals = np.ones((len(samples), len(shifts), len(dr_techniques_plot), len(od_tests), len(features), random_runs)) * (-1)
    samples_shifts_rands_feat_t_vals = np.ones((len(samples), len(shifts), len(dr_techniques_plot), len(od_tests), len(features), random_runs)) * (-1)

    for shift_idx, shift in enumerate(shifts):

        shift_path =  hosp_path + shift + '/'
        if not os.path.exists(shift_path):
            os.makedirs(shift_path)

        rand_run_p_vals = np.ones((len(samples), len(dr_techniques_plot), random_runs)) * (-1)
        rand_run_t_vals = np.ones((len(samples), len(dr_techniques_plot), random_runs)) * (-1)

        # Stores accuracy values for malignancy detection.
        rand_run_tr_auc = np.ones((len(samples), random_runs)) * (-1)
        rand_run_te_auc = np.ones((len(samples), random_runs)) * (-1)
        rand_run_tr_smr = np.ones((len(samples), random_runs)) * (-1)
        rand_run_te_smr = np.ones((len(samples), random_runs)) * (-1)


        rand_run_tr_eo = np.ones((len(samples), random_runs)) * (-1)
        rand_run_tr_dp = np.ones((len(samples), random_runs)) * (-1)
        rand_run_val_eo = np.ones((len(samples), random_runs)) * (-1)
        rand_run_val_dp = np.ones((len(samples), random_runs)) * (-1)
        rand_run_te_eo = np.ones((len(samples), random_runs)) * (-1)
        rand_run_te_dp = np.ones((len(samples), random_runs)) * (-1)

        rand_run_feat_p_vals = np.ones((len(samples), len(dr_techniques_plot), len(od_tests), len(features), random_runs)) * (-1)
        rand_run_feat_t_vals = np.ones((len(samples), len(dr_techniques_plot), len(od_tests), len(features), random_runs)) * (-1)

        for rand_run in range(random_runs):

            print("\nRandom run %s" % rand_run)

            rand_run_path = shift_path + str(rand_run) + '/'
            if not os.path.exists(rand_run_path):
                os.makedirs(rand_run_path)

            np.random.seed(rand_run)
            tensorflow.random.set_seed(rand_run)

            # Load data
            time_now = time.time()
            # print('Original')
            (X_tr_orig, y_tr_orig, sens_tr_orig), (X_val_orig, y_val_orig, sens_val_orig), (X_te_orig, y_te_orig, sens_te_orig), orig_dims, nb_classes = load_hosp_dataset(datset, df, target, features, hosp_train, hosp_test, use_group, shuffle=False)
            # X_tr_orig = normalize_datapoints(X_tr_orig, 255.)
            # X_te_orig = normalize_datapoints(X_te_orig, 255.)
            # X_val_orig = normalize_datapoints(X_val_orig, 255.)
            X_te_1 = X_te_orig.copy()
            y_te_1 = y_te_orig.copy()
            sens_te_1 = sens_te_orig.copy()

            # Apply shift
            if shift != 'orig':
                (X_tr_orig, y_tr_orig, sens_tr_orig), (X_val_orig, y_val_orig, sens_val_orig), (X_te_orig, y_te_orig, sens_te_orig), orig_dims, nb_classes = load_hosp_dataset(datset, df, target, features, hosp_train, hosp_test, use_group, shuffle=True)
                # X_tr_orig = normalize_datapoints(X_tr_orig, 255.)
                # X_te_orig = normalize_datapoints(X_te_orig, 255.)
                # X_val_orig = normalize_datapoints(X_val_orig, 255.)
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
                shift_detector = ShiftDetector(dr_techniques, test_types, od_tests, md_tests, sign_level, red_models, sample, datset)
                (od_decs, ind_od_decs, ind_od_p_vals, ind_od_t_vals, ind_od_feat_p_vals, ind_od_feat_t_vals),\
                (md_decs, ind_md_decs, ind_md_p_vals, ind_md_t_vals),\
                red_dim, red_models, tr_auc, te_val_auc_diff, tr_smr, te_val_smr_diff,\
                tr_eo, tr_dp, val_eo, val_dp, te_eo, te_dp\
                 = shift_detector.detect_data_shift(X_tr_3, y_tr_3, sens_tr_3, X_val_3, y_val_3, sens_val_3, X_te_3, y_te_3, sens_te_3, orig_dims, nb_classes)

                time_diff = time.time() - time_now
                print('Time: {}s taken in detecting shift\n{}, {}, {}'.format(time_diff, rand_run, hosp_train, hosp_test))

                rand_run_tr_auc[si, rand_run] = tr_auc
                rand_run_te_auc[si, rand_run] = te_val_auc_diff
                rand_run_tr_smr[si, rand_run] = tr_smr
                rand_run_te_smr[si, rand_run] = te_val_smr_diff
                rand_run_tr_eo[si, rand_run] = tr_eo
                rand_run_tr_dp[si, rand_run] = tr_dp
                rand_run_val_eo[si, rand_run] = val_eo
                rand_run_val_dp[si, rand_run] = val_dp
                rand_run_te_eo[si, rand_run] = te_eo
                rand_run_te_dp[si, rand_run] = te_dp

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

                    # if datset == 'mnist' or datset == 'mnist_usps' or datset == 'mnist_usps':
                    #     samp_shape = (28,28)
                    #     cmap = 'gray'
                    # elif datset == 'cifar10' or datset == 'svhn':
                    #     samp_shape = (32,32,3)
                    #     cmap = None
                    # elif datset == 'eicu':
                    #     samp_shape = (1,X_tr_orig.shape[1]) # TODO change feature representation
                    #     cmap = 'gray'
                    
                    # if dec:
                    #     most_conf_test_indices = test_indices[test_perc > 0.8]

                    #     top_same_samples_path = sample_path + 'top_same'
                    #     if not os.path.exists(top_same_samples_path):
                    #         os.makedirs(top_same_samples_path)

                    #     rev_top_test_ind = test_indices[::-1][:difference_samples]
                    #     least_conf_samples = X_te_dcl[rev_top_test_ind]
                    #     for j in range(len(rev_top_test_ind)):
                    #         samp = least_conf_samples[j, :]
                    #         fig = plt.imshow(samp.reshape(samp_shape), cmap=cmap)
                    #         plt.axis('off')
                    #         fig.axes.get_xaxis().set_visible(False)
                    #         fig.axes.get_yaxis().set_visible(False)
                    #         plt.savefig("%s/%s.pdf" % (top_same_samples_path, j), bbox_inches='tight', pad_inches=0)
                    #         plt.clf()

                    #         j = j + 1

                    #     top_different_samples_path = sample_path + 'top_diff'
                    #     if not os.path.exists(top_different_samples_path):
                    #         os.makedirs(top_different_samples_path)

                    #     most_conf_samples = X_te_dcl[most_conf_test_indices]
                    #     original_indices = []
                    #     j = 0
                    #     for i in range(len(most_conf_samples)):
                    #         samp = most_conf_samples[i,:]
                    #         ind = np.where(np.all(X_te_3==samp,axis=1))
                    #         if len(ind[0]) > 0:
                    #             # original_indices.append(np.asscalar(ind[0])) # TODO: handle len(ind[0])>1 i.e. cases where >1 indices match X_te_3==samp

                    #             if j < difference_samples:
                    #                 fig = plt.imshow(samp.reshape(samp_shape), cmap=cmap)
                    #                 plt.axis('off')
                    #                 fig.axes.get_xaxis().set_visible(False)
                    #                 fig.axes.get_yaxis().set_visible(False)
                    #                 plt.savefig("%s/%s.pdf" % (top_different_samples_path,j), bbox_inches='tight', pad_inches = 0)
                    #                 plt.clf()

                    #                 j = j + 1

            # for dr_idx, dr in enumerate(dr_techniques_plot):
            #     plt.semilogx(np.array(samples), rand_run_p_vals[:,dr_idx,rand_run], format[dr], color=colors[dr], label="%s" % DimensionalityReduction(dr).name)
            # plt.axhline(y=sign_level, color='k')
            # plt.xlabel('Number of samples from test')
            # plt.ylabel('$p$-value')
            # plt.savefig("%s/dr_sample_comp_noleg.pdf" % rand_run_path, bbox_inches='tight')
            # plt.legend()
            # plt.savefig("%s/dr_sample_comp.pdf" % rand_run_path, bbox_inches='tight')
            # plt.clf()

            np.savetxt("%s/dr_method_p_vals.csv" % rand_run_path, rand_run_p_vals[:,:,rand_run], delimiter=",")
            np.savetxt("%s/dr_method_t_vals.csv" % rand_run_path, rand_run_t_vals[:,:,rand_run], delimiter=",")

            np.save("%s/dr_method_feat_p_vals.npy" % rand_run_path, rand_run_feat_p_vals[:,:,:,:,rand_run])
            np.save("%s/dr_method_feat_t_vals.npy" % rand_run_path, rand_run_feat_t_vals[:,:,:,:,rand_run])


        mean_p_vals = np.mean(rand_run_p_vals, axis=2)
        std_p_vals = np.std(rand_run_p_vals, axis=2)
        
        mean_te_auc = np.mean(rand_run_te_auc, axis=1)
        std_te_auc = np.std(rand_run_te_auc, axis=1)
        mean_te_smr = np.mean(rand_run_te_smr, axis=1)
        std_te_smr = np.std(rand_run_te_smr, axis=1)
        mean_te_eo = np.mean(rand_run_te_eo, axis=1)
        std_te_eo = np.std(rand_run_te_eo, axis=1)
        mean_te_dp = np.mean(rand_run_te_dp, axis=1)
        std_te_dp = np.std(rand_run_te_dp, axis=1)
        
        mean_feat_p_vals = np.mean(rand_run_feat_p_vals, axis=4)
        std_feat_p_vals = np.std(rand_run_feat_p_vals, axis=4)
        # for dr_idx, dr in enumerate(dr_techniques_plot):
        #     errorfill(np.array(samples), mean_p_vals[:,dr_idx], std_p_vals[:,dr_idx], fmt=format[dr], color=colors[dr], label="%s" % DimensionalityReduction(dr).name)
        # plt.axhline(y=sign_level, color='k')
        # plt.xlabel('Number of samples from test')
        # plt.ylabel('$p$-value')
        # plt.savefig("%s/dr_sample_comp_noleg.pdf" % shift_path, bbox_inches='tight')
        # plt.legend()
        # plt.savefig("%s/dr_sample_comp.pdf" % shift_path, bbox_inches='tight')
        # plt.clf()

        # for dr_idx, dr in enumerate(dr_techniques_plot):
        #     errorfill(np.array(samples), mean_p_vals[:,dr_idx], std_p_vals[:,dr_idx], fmt=format[dr], color=colors[dr])
        #     plt.xlabel('Number of samples from test')
        #     plt.ylabel('$p$-value')
        #     plt.axhline(y=sign_level, color='k', label='sign_level')
        #     plt.savefig("%s/%s_conf.pdf" % (shift_path, DimensionalityReduction(dr).name), bbox_inches='tight')
        #     plt.clf()

        np.savetxt("%s/mean_p_vals.csv" % shift_path, mean_p_vals, delimiter=",")
        np.savetxt("%s/std_p_vals.csv" % shift_path, std_p_vals, delimiter=",")
        np.savetxt("%s/mean_te_auc.csv" % shift_path, mean_te_auc, delimiter=",")
        np.savetxt("%s/std_te_auc.csv" % shift_path, std_te_auc, delimiter=",")
        np.savetxt("%s/mean_te_smr.csv" % shift_path, mean_te_smr, delimiter=",")
        np.savetxt("%s/std_te_smr.csv" % shift_path, std_te_smr, delimiter=",")
        np.savetxt("%s/mean_te_eo.csv" % shift_path, mean_te_eo, delimiter=",")
        np.savetxt("%s/std_te_eo.csv" % shift_path, std_te_eo, delimiter=",")
        np.savetxt("%s/mean_te_dp.csv" % shift_path, mean_te_dp, delimiter=",")
        np.savetxt("%s/std_te_dp.csv" % shift_path, std_te_dp, delimiter=",")

        np.save("%s/mean_feat_p_vals.npy" % shift_path, mean_feat_p_vals)
        np.save("%s/std_feat_p_vals.npy" % shift_path, std_feat_p_vals)

        for dr_idx, dr in enumerate(dr_techniques_plot):
            samples_shifts_rands_dr_tech[:,shift_idx,:,dr_idx] = rand_run_p_vals[:,dr_idx,:]
            samples_shifts_rands_dr_tech_t_val[:,shift_idx,:,dr_idx] = rand_run_t_vals[:,dr_idx,:]

            samples_shifts_rands_feat_p_vals[:,shift_idx,dr_idx,:,:,:] = rand_run_feat_p_vals[:,dr_idx,:,:,:]
            samples_shifts_rands_feat_t_vals[:,shift_idx,dr_idx,:,:,:] = rand_run_feat_t_vals[:,dr_idx,:,:,:]

        np.save("%s/samples_shifts_rands_dr_tech.npy" % (hosp_path), samples_shifts_rands_dr_tech)
        np.save("%s/samples_shifts_rands_dr_tech_t_val.npy" % (hosp_path), samples_shifts_rands_dr_tech_t_val)

        samples_shifts_rands_te_acc[:,shift_idx,:,0] = rand_run_te_auc
        samples_shifts_rands_te_acc[:,shift_idx,:,1] = rand_run_te_smr
        samples_shifts_rands_te_acc[:,shift_idx,:,2] = rand_run_te_eo
        samples_shifts_rands_te_acc[:,shift_idx,:,3] = rand_run_te_dp

        np.save("%s/samples_shifts_rands_te_acc.npy" % (hosp_path), samples_shifts_rands_te_acc)

        np.save("%s/samples_shifts_rands_feat_p_vals.npy" % (hosp_path), samples_shifts_rands_feat_p_vals)
        np.save("%s/samples_shifts_rands_feat_t_vals.npy" % (hosp_path), samples_shifts_rands_feat_t_vals)

    # for dr_idx, dr in enumerate(dr_techniques_plot):
    #     dr_method_results = samples_shifts_rands_dr_tech[:,:,:,dr_idx]

    #     mean_p_vals = np.mean(dr_method_results, axis=2)
    #     std_p_vals = np.std(dr_method_results, axis=2)

        # for idx, shift in enumerate(shifts):
        #     errorfill(np.array(samples), mean_p_vals[:, idx], std_p_vals[:, idx], fmt=linestyles[idx]+markers[dr], color=colorscale(colors[dr],brightness[idx]), label="%s" % shift.replace('_', '\\_'))
        # plt.xlabel('Number of samples from test')
        # plt.ylabel('$p$-value')
        # plt.axhline(y=sign_level, color='k')
        # plt.savefig("%s/%s_conf_noleg.pdf" % (hosp_path, DimensionalityReduction(dr).name), bbox_inches='tight')
        # plt.legend()
        # plt.savefig("%s/%s_conf.pdf" % (hosp_path, DimensionalityReduction(dr).name), bbox_inches='tight')
        # plt.clf()

    np.save("%s/samples_shifts_rands_dr_tech.npy" % (hosp_path), samples_shifts_rands_dr_tech)
    np.save("%s/samples_shifts_rands_dr_tech_t_val.npy" % (hosp_path), samples_shifts_rands_dr_tech_t_val)
    np.save("%s/samples_shifts_rands_te_acc.npy" % (hosp_path), samples_shifts_rands_te_acc)

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
                                            hosp_train, hosp_test, use_group) for (hosp_train, hosp_test) in hosp_pairs)