'''
Written by Stephan Rabanser https://github.com/steverab/failing-loudly
Modifed

Detect shifts across hospitals

Usage:
python hosp_pipeline_parallel.py eicu orig multiv mice

# TODO
mice, acc BBSDh, univariate
add gcs in data file
mean_p_vals = -1 for 73, 338

impute missing values in shift_reductor pca, srp, lda
number of dims in shift_detector
shift_tester.test_shift one dim check if t_val correct after FWER correction
shift_tester.test_chi2_shift one dim return t_val
use validation set
load data once
'''

import numpy as np
from tensorflow import set_random_seed
seed = 1
np.random.seed(seed)
set_random_seed(seed)

from shift_detector import *
from shift_locator import *
from shift_applicator import *
from data_utils import *
import os
import sys
from exp_utils import *

import multiprocessing
from joblib import Parallel, delayed
num_cores = min(41, multiprocessing.cpu_count())

# -------------------------------------------------
# PLOTTING HELPERS
# -------------------------------------------------

import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)
rc('axes', labelsize=20)
rc('xtick', labelsize=20)
rc('ytick', labelsize=20)
rc('legend', fontsize=12)

def clamp(val, minimum=0, maximum=255):
    if val < minimum:
        return minimum
    if val > maximum:
        return maximum
    return val

def colorscale(hexstr, scalefactor):
    hexstr = hexstr.strip('#')

    if scalefactor < 0 or len(hexstr) != 6:
        return hexstr

    r, g, b = int(hexstr[:2], 16), int(hexstr[2:4], 16), int(hexstr[4:], 16)

    r = clamp(r * scalefactor)
    g = clamp(g * scalefactor)
    b = clamp(b * scalefactor)

    return "#%02x%02x%02x" % (int(r), int(g), int(b))

linestyles = ['-', '-.', '--', ':']
brightness = [1.25, 1.0, 0.75, 0.5]
format = ['-o', '-h', '-p', '-s', '-D', '-<', '->', '-X']
markers = ['o', 'h', 'p', 's', 'D', '<', '>', 'X']
colors_old = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']
colors = ['#2196f3', '#f44336', '#9c27b0', '#64dd17', '#009688', '#ff9800', '#795548', '#607d8b']

def errorfill(x, y, yerr, color=None, alpha_fill=0.2, ax=None, fmt='-o', label=None):
    ax = ax if ax is not None else plt.gca()
    if color is None:
        color = next(ax._get_lines.prop_cycler)['color']
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    ax.semilogx(x, y, fmt, color=color, label=label)
    ax.fill_between(x, np.clip(ymax, 0, 1), np.clip(ymin, 0, 1), color=color, alpha=alpha_fill)

# -------------------------------------------------
# CONFIG
# -------------------------------------------------

# make_keras_picklable()
np.set_printoptions(threshold=sys.maxsize)

datset = sys.argv[1]
test_type = sys.argv[3]
missing_imp = sys.argv[4]

path = './hosp_results_parallel/'
path += test_type + '/'
path += datset + '_'
path += sys.argv[2] + '/'

if not os.path.exists(path):
    os.makedirs(path)

# Define feature groups
# feature_sets = [['labs','vitals','demo','others']]
# feature_sets = [['labs']]
# feature_sets = [['vitals']]
# feature_sets = [['demo']]
feature_sets = [['labs','vitals','demo','others'], ['labs'], ['vitals'], ['demo']]

# Define train-test pairs of hospitals 
hosp_pairs = []
HospitalIDs = HospitalIDs[:11]
for hi in HospitalIDs:
    for hj in HospitalIDs:
        hosp_pairs.append(([hi],[hj]))
# hosp_pairs = [([394],[416])]

# Define DR methods
# dr_techniques = [DimensionalityReduction.NoRed.value, DimensionalityReduction.PCA.value, DimensionalityReduction.SRP.value, DimensionalityReduction.UAE.value, DimensionalityReduction.TAE.value, DimensionalityReduction.BBSDs.value, DimensionalityReduction.BBSDh.value]
dr_techniques = [DimensionalityReduction.NoRed.value, DimensionalityReduction.PCA.value]
# dr_techniques = [DimensionalityReduction.NoRed.value, DimensionalityReduction.PCA.value, DimensionalityReduction.SRP.value]
if test_type == 'multiv':
    # dr_techniques = [DimensionalityReduction.NoRed.value, DimensionalityReduction.PCA.value, DimensionalityReduction.SRP.value, DimensionalityReduction.UAE.value, DimensionalityReduction.TAE.value, DimensionalityReduction.BBSDs.value]
    dr_techniques = [DimensionalityReduction.NoRed.value, DimensionalityReduction.PCA.value]
    # dr_techniques = [DimensionalityReduction.NoRed.value, DimensionalityReduction.PCA.value, DimensionalityReduction.SRP.value]
if test_type == 'univ':
    dr_techniques_plot = dr_techniques.copy()
    dr_techniques_plot.append(DimensionalityReduction.Classif.value)
else:
    dr_techniques_plot = dr_techniques.copy()

# Define test types and general test sample sizes
test_types = [td.value for td in TestDimensionality]
if test_type == 'multiv':
    od_tests = []
    md_tests = [MultidimensionalTest.MMD.value]
    # samples = [10, 20, 50, 100, 200, 500, 1000]
    # samples = [100, 1000]
    samples = [1000]
    # samples = [10, 20, 50, 100, 200]
else:
    # od_tests = [od.value for od in OnedimensionalTest]
    od_tests = [OnedimensionalTest.KS.value]
    md_tests = []
    # samples = [10, 20, 50, 100, 200, 500, 1000, 9000]
    # samples = [100, 1000]
    samples = [1000]
    # samples = [10, 20, 50, 100, 200, 500]
difference_samples = 10

if missing_imp == 'mice':
    missing_techniques = ['org', 'mice']
else:
    missing_techniques = ['org']

# Number of random runs to average results over    
random_runs = 5

# Signifiance level
sign_level = 0.05

# Define shift types
if sys.argv[2] == 'orig':
    shifts = ['orig']
    brightness = [0.75]
    # shifts = ['rand', 'orig']
    # brightness = [1.25, 0.75]
else:
    shifts = []

# -------------------------------------------------
# PIPELINE START
# -------------------------------------------------

def test_hosp_pair(feature_set_idx, feature_set, hosp_pair_idx, hosp_train, hosp_test):
    print("\n========\nFeature Set, Hosp Train, Hosp Test", feature_set, hosp_train, hosp_test)
    print("========\n")

    hosp_folder_name = 'tr_' + '_'.join(map(str, hosp_train)) + '_ts_' + '_'.join(map(str, hosp_test))
    hosp_path = path + "_".join(feature_set) + '/' + hosp_folder_name + '/'
    if not os.path.exists(hosp_path):
        os.makedirs(hosp_path)

    samples_shifts_rands_dr_tech = np.ones((len(samples), len(shifts), random_runs, len(dr_techniques) + 1)) * (-1) # TODO add hosp_pair
    samples_shifts_rands_dr_tech_t_val = np.ones((len(samples), len(shifts), random_runs, len(dr_techniques) + 1)) * (-1) # TODO add hosp_pair

    red_dim = -1
    red_models = [None] * len(DimensionalityReduction)

    for shift_idx, shift in enumerate(shifts):

        shift_path =  hosp_path + shift + '/'
        if not os.path.exists(shift_path):
            os.makedirs(shift_path)

        rand_run_p_vals = np.ones((len(samples), len(dr_techniques) + 1, random_runs)) * (-1)
        rand_run_t_vals = np.ones((len(samples), len(dr_techniques) + 1, random_runs)) * (-1)

        for rand_run in range(random_runs):

            print("\nRandom run %s" % rand_run)

            rand_run_path = shift_path + str(rand_run) + '/'
            if not os.path.exists(rand_run_path):
                os.makedirs(rand_run_path)

            np.random.seed(rand_run)
            set_random_seed(rand_run)

            # Load data
            (X_tr_orig, y_tr_orig), (X_val_orig, y_val_orig), (X_te_orig, y_te_orig), orig_dims, nb_classes = import_hosp_dataset(datset, feature_set, hosp_train, hosp_test, shuffle=True)
            # X_tr_orig = normalize_datapoints(X_tr_orig, 255.)
            # X_te_orig = normalize_datapoints(X_te_orig, 255.)
            # X_val_orig = normalize_datapoints(X_val_orig, 255.)

            # Apply shift
            if shift == 'orig':
                # print('Original')
                (X_tr_orig, y_tr_orig), (X_val_orig, y_val_orig), (X_te_orig, y_te_orig), orig_dims, nb_classes = import_hosp_dataset(datset, feature_set, hosp_train, hosp_test)
                # X_tr_orig = normalize_datapoints(X_tr_orig, 255.)
                # X_te_orig = normalize_datapoints(X_te_orig, 255.)
                # X_val_orig = normalize_datapoints(X_val_orig, 255.)
                X_te_1 = X_te_orig.copy()
                y_te_1 = y_te_orig.copy()
            else:
                (X_te_1, y_te_1) = apply_shift(X_te_orig, y_te_orig, shift, orig_dims, datset)

            X_te_2 , y_te_2 = random_shuffle(X_te_1, y_te_1)

            # Check detection performance for different numbers of samples from test
            for si, sample in enumerate(samples):

                # print("Sample %s" % sample)

                sample_path = rand_run_path + str(sample) + '/'
                if not os.path.exists(sample_path):
                    os.makedirs(sample_path)

                X_te_3 = X_te_2[:sample,:]
                x_te_3_samp = X_te_3[0]
                y_te_3 = y_te_2[:sample]

                if test_type == 'multiv':
                    X_val_3 = X_val_orig[:1000,:]
                    y_val_3 = y_val_orig[:1000]
                else:
                    X_val_3 = np.copy(X_val_orig)
                    y_val_3 = np.copy(y_val_orig)

                X_tr_3 = np.copy(X_tr_orig)
                y_tr_3 = np.copy(y_tr_orig)

                # Detect shift
                shift_detector = ShiftDetector(dr_techniques, test_types, od_tests, md_tests, sign_level, red_models, sample, datset)
                (od_decs, ind_od_decs, ind_od_p_vals, ind_od_t_vals), (md_decs, ind_md_decs, ind_md_p_vals, ind_md_t_vals), red_dim, red_models, val_acc, te_acc = shift_detector.detect_data_shift(X_tr_3, y_tr_3, X_val_3, y_val_3, X_te_3, y_te_3, orig_dims, nb_classes)

                if test_type == 'multiv':
                    # print("Shift decision: ", ind_md_decs.flatten())
                    # print("Shift p-vals: ", ind_md_p_vals.flatten())

                    rand_run_p_vals[si,:,rand_run] = np.append(ind_md_p_vals.flatten(), -1) # no value for Classif dr method # TODO: reduce size
                    rand_run_t_vals[si,:,rand_run] = np.append(ind_md_t_vals.flatten(), -1) # no value for Classif dr method # TODO: reduce size
                else:
                    # print("Shift decision: ", ind_od_decs.flatten())
                    # print("Shift p-vals: ", ind_od_p_vals.flatten())

                    # Characterize shift via difference classifier
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


        mean_p_vals = np.mean(rand_run_p_vals, axis=2)
        std_p_vals = np.std(rand_run_p_vals, axis=2)

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

        for dr_idx, dr in enumerate(dr_techniques_plot):
            samples_shifts_rands_dr_tech[:,shift_idx,:,dr_idx] = rand_run_p_vals[:,dr_idx,:]
            samples_shifts_rands_dr_tech_t_val[:,shift_idx,:,dr_idx] = rand_run_t_vals[:,dr_idx,:]

        np.save("%s/samples_shifts_rands_dr_tech.npy" % (hosp_path), samples_shifts_rands_dr_tech)
        np.save("%s/samples_shifts_rands_dr_tech_t_val.npy" % (hosp_path), samples_shifts_rands_dr_tech_t_val)


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

for feature_set_idx, feature_set in enumerate(feature_sets):

    Parallel(n_jobs=num_cores)(delayed(test_hosp_pair)(feature_set_idx, feature_set,\
                                        hosp_pair_idx, hosp_train, hosp_test) for hosp_pair_idx, (hosp_train, hosp_test) in enumerate(hosp_pairs))