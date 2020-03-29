'''
Written by Stephan Rabanser https://github.com/steverab/failing-loudly
Modifed

Usage:
python generate_hosp_plot.py eicu orig multiv

Plot test results across hospitals
'''

import numpy as np
import seaborn as sns
import pandas as pd
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

path = './hosp_results_parallel/'
path += test_type + '/'
path += datset + '_'
path += sys.argv[2] + '/'

if not os.path.exists(path):
    os.makedirs(path)

# Define feature groups
# feature_sets = [['labs','vitals','demo','others']]
# feature_sets = [['labs','vitals','demo','others'], ['labs']]
# feature_sets = [['labs','vitals','demo','others'], ['labs'], ['vitals']]
feature_sets = [['labs','vitals','demo','others'], ['labs'], ['vitals'], ['demo']]

# Define train-test pairs of hospitals 
hosp_pairs = []
HospitalIDs = HospitalIDs[:11]
# HospitalIDs = [i for i in HospitalIDs if i not in [413,394,199,345]]
for hi in range(len(HospitalIDs)):
    for hj in range(len(HospitalIDs)):
        hosp_pairs.append((hi,hj,[HospitalIDs[hi]],[HospitalIDs[hj]]))
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

samples_shifts_rands_dr_tech_feats_hosps = np.ones((len(samples), len(shifts), random_runs, len(dr_techniques) + 1, len(feature_sets), len(hosp_pairs))) * (-1)
samples_shifts_rands_dr_tech_feats_hosps_t_val = np.ones((len(samples), len(shifts), random_runs, len(dr_techniques) + 1, len(feature_sets), len(hosp_pairs))) * (-1)

for feature_set_idx, feature_set in enumerate(feature_sets):

    for hosp_pair_idx, (_, _, hosp_train, hosp_test) in enumerate(hosp_pairs):
    
        print("\n==========\nFeature Set, Hosp Train, Hosp Test", feature_set, hosp_train, hosp_test)
        print("==========\n")

        feats_path = path + "_".join(feature_set) + '/'
        hosp_folder_name = 'tr_' + '_'.join(map(str, hosp_train)) + '_ts_' + '_'.join(map(str, hosp_test))
        hosp_path = feats_path + hosp_folder_name + '/'
        
        samples_shifts_rands_dr_tech = np.load("%s/samples_shifts_rands_dr_tech.npy" % (hosp_path))
        samples_shifts_rands_dr_tech_t_val = np.load("%s/samples_shifts_rands_dr_tech_t_val.npy" % (hosp_path))

        samples_shifts_rands_dr_tech_feats_hosps[:,:,:,:,feature_set_idx,hosp_pair_idx] = samples_shifts_rands_dr_tech
        samples_shifts_rands_dr_tech_feats_hosps_t_val[:,:,:,:,feature_set_idx,hosp_pair_idx] = samples_shifts_rands_dr_tech_t_val

np.save("%s/samples_shifts_rands_dr_tech_feats_hosps.npy" % (path), samples_shifts_rands_dr_tech_feats_hosps)
np.save("%s/samples_shifts_rands_dr_tech_feats_hosps_t_val.npy" % (path), samples_shifts_rands_dr_tech_feats_hosps_t_val)

# Feat, dr, shift, sample - mean
for feature_set_idx, feature_set in enumerate(feature_sets):

    feats_path = path + "_".join(feature_set) + '/'

    for dr_idx, dr in enumerate(dr_techniques_plot):

        for shift_idx, shift in enumerate(shifts):

            for si, sample in enumerate(samples):

                hosp_pair_pval = np.ones((len(HospitalIDs), len(HospitalIDs))) * (-1)
                hosp_pair_tval = np.ones((len(HospitalIDs), len(HospitalIDs))) * (-1)
                for hosp_pair_idx, (hosp_train_idx, hosp_test_idx, hosp_train, hosp_test) in enumerate(hosp_pairs):
            
                    feats_dr_tech_shifts_samples_results = samples_shifts_rands_dr_tech_feats_hosps[si,shift_idx,:,dr_idx,feature_set_idx,hosp_pair_idx]
                    feats_dr_tech_shifts_samples_results_t_val = samples_shifts_rands_dr_tech_feats_hosps_t_val[si,shift_idx,:,dr_idx,feature_set_idx,hosp_pair_idx]

                    mean_p_vals = np.mean(feats_dr_tech_shifts_samples_results)
                    std_p_vals = np.std(feats_dr_tech_shifts_samples_results)
                    mean_t_vals = np.mean(feats_dr_tech_shifts_samples_results_t_val)
                    # if mean_p_vals==-1:
                    #     print(hosp_train, hosp_test)
                    #     mean_p_vals = 1
                    hosp_pair_pval[hosp_train_idx, hosp_test_idx] = mean_p_vals
                    hosp_pair_tval[hosp_train_idx, hosp_test_idx] = mean_t_vals

                hosp_pair_pval = pd.DataFrame(hosp_pair_pval, columns=HospitalIDs, index=HospitalIDs)
                cmap = sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True)
                fig = sns.heatmap(hosp_pair_pval, linewidths=0.5, cmap=cmap)
                plt.xlabel('Target hospital')
                plt.ylabel('Source hospital')
                plt.savefig("%s/%s_%s_%s_p_val_hmp.pdf" % (feats_path, DimensionalityReduction(dr).name, shift, sample), bbox_inches='tight')
                plt.clf()

                hosp_pair_tval = pd.DataFrame(hosp_pair_tval, columns=HospitalIDs, index=HospitalIDs)
                fig = sns.heatmap(hosp_pair_tval, linewidths=0.5, cmap=cmap)
                plt.xlabel('Target hospital')
                plt.ylabel('Source hospital')
                plt.savefig("%s/%s_%s_%s_t_val_hmp.pdf" % (feats_path, DimensionalityReduction(dr).name, shift, sample), bbox_inches='tight')
                plt.clf()