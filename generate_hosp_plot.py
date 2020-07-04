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
from scipy import stats
from matplotlib.colors import ListedColormap
from itertools import combinations
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

path = './hosp_results_gossis/'
path += test_type + '/'
path += datset + '_'
path += sys.argv[2] + '/'

if not os.path.exists(path):
    os.makedirs(path)

# Define feature groups
# feature_groups = [['labs','vitals','demo','others','saps2diff']]
# feature_groups = [['labs','vitals','demo','others'], ['labs']]
# feature_groups = [['labs','vitals','demo','others'], ['labs'], ['vitals']]
# feature_groups = [['saps2']]
# feature_groups = [['saps2'], ['labs','vitals','demo','others']]
# feature_groups = [['labs'], ['vitals'], ['demo']]
# feature_groups = [['saps2'], ['labs','vitals','demo','others','saps2diff'], ['labs'], ['vitals'], ['demo']]
feature_groups = [['APACHE_covariate']]
# feature_groups = [['APACHE_covariate'], ['labs','labs_blood_gas'], ['vitals'], ['APACHE_comorbidity'],
#                     ['demographic','vitals','labs','labs_blood_gas','APACHE_comorbidity']]

# Define train-test pairs of hospitals 
NUM_HOSPITALS_TOP = 5 # hospitals with records >= 1000
hosp_pairs = []
# TODO move to data_utils
if datset =='eicu':
    HospitalIDs = HospitalIDs_eicu
    FeatureGroups = FeatureGroups_eicu
elif datset =='gossis':
    HospitalIDs = HospitalIDs_gossis
    FeatureGroups = FeatureGroups_gossis

HospitalIDs = HospitalIDs[:NUM_HOSPITALS_TOP]
# HospitalIDs = [i for i in HospitalIDs if i not in [413,394,199,345]]
for hi in range(len(HospitalIDs)):
    for hj in range(len(HospitalIDs)):
        hosp_pairs.append((hi,hj,[HospitalIDs[hi]],[HospitalIDs[hj]]))
# hosp_pairs = [([394],[416])]

# Define DR methods
# dr_techniques = [DimensionalityReduction.NoRed.value, DimensionalityReduction.PCA.value, DimensionalityReduction.SRP.value, DimensionalityReduction.UAE.value, DimensionalityReduction.TAE.value, DimensionalityReduction.BBSDs.value, DimensionalityReduction.BBSDh.value]
# dr_techniques = [DimensionalityReduction.NoRed.value]
dr_techniques = [DimensionalityReduction.NoRed.value, DimensionalityReduction.PCA.value]
# dr_techniques = [DimensionalityReduction.NoRed.value, DimensionalityReduction.PCA.value, DimensionalityReduction.SRP.value]
if test_type == 'multiv':
    # dr_techniques = [DimensionalityReduction.NoRed.value, DimensionalityReduction.PCA.value, DimensionalityReduction.SRP.value, DimensionalityReduction.UAE.value, DimensionalityReduction.TAE.value, DimensionalityReduction.BBSDs.value]
    dr_techniques = [DimensionalityReduction.NoRed.value, DimensionalityReduction.PCA.value]
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
    # samples = [10, 20, 50, 100, 200, 500, 1000]
    # samples = [100, 1000]
    samples = [1500]
    # samples = [1000, 1500]
    # samples = [10, 20, 50, 100, 200]
else:
    # od_tests = [od.value for od in OnedimensionalTest]
    od_tests = [OnedimensionalTest.KS.value]
    md_tests = []
    # samples = [10, 20, 50, 100, 200, 500, 1000, 9000]
    # samples = [100, 1000]
    samples = [1500]
    # samples = [1000, 1500]
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

sns.set_style("ticks")
cmap = sns.cubehelix_palette(2, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True)
# Discrete colormap using code by lanery https://stackoverflow.com/questions/38836154/discrete-legend-in-seaborn-heatmap-plot                
cmap_binary = sns.cubehelix_palette(2, hue=0.05, rot=0, light=0.9, dark=0)

samples_shifts_rands_dr_tech_feats_hosps = np.ones((len(samples), len(shifts), random_runs, len(dr_techniques_plot), len(feature_groups), len(hosp_pairs))) * (-1)
samples_shifts_rands_dr_tech_feats_hosps_t_val = np.ones((len(samples), len(shifts), random_runs, len(dr_techniques_plot), len(feature_groups), len(hosp_pairs))) * (-1)
samples_shifts_rands_feats_hosps_te_acc = np.ones((len(samples), len(shifts), random_runs, len(feature_groups), len(hosp_pairs), 2)) * (-1) # 0-auc, 1-smr # TODO add auc, smr, p-val, mmd in same array. add hosp_pair

for feature_group_idx, feature_group in enumerate(feature_groups):

    target = FeatureGroups['outcome']
    feature_set = []
    for group in feature_group:
        feature_set += FeatureGroups[group]

    samples_shifts_rands_feat_hosps_p_vals = np.ones((len(samples), len(shifts), len(dr_techniques_plot), len(od_tests), len(feature_set), random_runs, len(hosp_pairs))) * (-1)
    samples_shifts_rands_feat_hosps_t_vals = np.ones((len(samples), len(shifts), len(dr_techniques_plot), len(od_tests), len(feature_set), random_runs, len(hosp_pairs))) * (-1)

    for hosp_pair_idx, (_, _, hosp_train, hosp_test) in enumerate(hosp_pairs):

        print("\n==========\nFeature Set, Hosp Train, Hosp Test", feature_group, hosp_train, hosp_test)
        print("==========\n")

        feats_path = path + "_".join(feature_group) + '/'
        hosp_folder_name = 'tr_' + '_'.join(map(str, hosp_train)) + '_ts_' + '_'.join(map(str, hosp_test))
        hosp_path = feats_path + hosp_folder_name + '/'
        
        samples_shifts_rands_dr_tech = np.load("%s/samples_shifts_rands_dr_tech.npy" % (hosp_path))
        samples_shifts_rands_dr_tech_t_val = np.load("%s/samples_shifts_rands_dr_tech_t_val.npy" % (hosp_path))

        samples_shifts_rands_te_acc = np.load("%s/samples_shifts_rands_te_acc.npy" % (hosp_path))

        samples_shifts_rands_dr_tech_feats_hosps[:,:,:,:,feature_group_idx,hosp_pair_idx] = samples_shifts_rands_dr_tech
        samples_shifts_rands_dr_tech_feats_hosps_t_val[:,:,:,:,feature_group_idx,hosp_pair_idx] = samples_shifts_rands_dr_tech_t_val

        samples_shifts_rands_feats_hosps_te_acc[:,:,:,feature_group_idx,hosp_pair_idx,:] = samples_shifts_rands_te_acc

        if test_type == 'univ':
            samples_shifts_rands_feat_p_vals = np.load("%s/samples_shifts_rands_feat_p_vals.npy" % (hosp_path))
            samples_shifts_rands_feat_t_vals = np.load("%s/samples_shifts_rands_feat_t_vals.npy" % (hosp_path))

            samples_shifts_rands_feat_hosps_p_vals[:,:,:,:,:,:,hosp_pair_idx] = samples_shifts_rands_feat_p_vals
            samples_shifts_rands_feat_hosps_t_vals[:,:,:,:,:,:,hosp_pair_idx] = samples_shifts_rands_feat_t_vals
            
            np.save("%s/samples_shifts_rands_feat_hosps_p_vals.npy" % (feats_path), samples_shifts_rands_feat_hosps_p_vals)
            np.save("%s/samples_shifts_rands_feat_hosps_t_vals.npy" % (feats_path), samples_shifts_rands_feat_hosps_t_vals)

np.save("%s/samples_shifts_rands_dr_tech_feats_hosps.npy" % (path), samples_shifts_rands_dr_tech_feats_hosps)
np.save("%s/samples_shifts_rands_dr_tech_feats_hosps_t_val.npy" % (path), samples_shifts_rands_dr_tech_feats_hosps_t_val)
np.save("%s/samples_shifts_rands_feats_hosps_te_acc.npy" % (path), samples_shifts_rands_feats_hosps_te_acc)

# Feat, dr, shift, sample - mean
for feature_group_idx, feature_group in enumerate(feature_groups):

    target = FeatureGroups['outcome']
    feature_set = []
    for group in feature_group:
        feature_set += FeatureGroups[group]

    feats_path = path + "_".join(feature_group) + '/'

    if test_type == 'univ':
        samples_shifts_rands_feat_hosps_p_vals = np.load("%s/samples_shifts_rands_feat_hosps_p_vals.npy" % (feats_path))
        samples_shifts_rands_feat_hosps_t_vals = np.load("%s/samples_shifts_rands_feat_hosps_t_vals.npy" % (feats_path))

    for dr_idx, dr in enumerate(dr_techniques_plot):

        for shift_idx, shift in enumerate(shifts):

            for si, sample in enumerate(samples):

                hosp_pair_pval = np.ones((len(HospitalIDs), len(HospitalIDs))) * (-1)
                hosp_pair_tval = np.ones((len(HospitalIDs), len(HospitalIDs))) * (-1)

                hosp_pair_auc = np.ones((len(HospitalIDs), len(HospitalIDs))) * (-1)
                hosp_pair_smr = np.ones((len(HospitalIDs), len(HospitalIDs))) * (-1)

                if test_type == 'univ':
                    # hosp_pair_feat_pval = np.ones((len(hosp_pairs), len(feature_set), random_runs))
                    hosp_pair_feat_pval = np.ones((len(hosp_pairs), len(feature_set)))
                    hosp_pair_feat_tval = np.ones((len(hosp_pairs), len(feature_set)))

                for hosp_pair_idx, (hosp_train_idx, hosp_test_idx, hosp_train, hosp_test) in enumerate(hosp_pairs):
                    feats_dr_tech_shifts_samples_results = samples_shifts_rands_dr_tech_feats_hosps[si,shift_idx,:,dr_idx,feature_group_idx,hosp_pair_idx]
                    feats_dr_tech_shifts_samples_results_t_val = samples_shifts_rands_dr_tech_feats_hosps_t_val[si,shift_idx,:,dr_idx,feature_group_idx,hosp_pair_idx]
                    
                    mean_p_vals = np.mean(feats_dr_tech_shifts_samples_results)
                    std_p_vals = np.std(feats_dr_tech_shifts_samples_results)
                    mean_t_vals = np.mean(feats_dr_tech_shifts_samples_results_t_val)

                    hosp_pair_pval[hosp_train_idx, hosp_test_idx] = mean_p_vals < sign_level
                    hosp_pair_tval[hosp_train_idx, hosp_test_idx] = mean_t_vals

                    adjust_sign_level = sign_level / len(hosp_pairs)
                    if test_type == 'univ':
                        dr_tech_shifts_samples_results_feat_p_val = samples_shifts_rands_feat_hosps_p_vals[si,shift_idx,dr_idx,0,:,:,hosp_pair_idx] # TODO iterate for od_tests
                        dr_tech_shifts_samples_results_feat_t_val = samples_shifts_rands_feat_hosps_t_vals[si,shift_idx,dr_idx,0,:,:,hosp_pair_idx] # TODO iterate for od_tests

                        mean_feat_p_vals = np.mean(dr_tech_shifts_samples_results_feat_p_val, axis=1)
                        mean_feat_t_vals = np.mean(dr_tech_shifts_samples_results_feat_t_val, axis=1)

                        # hosp_pair_feat_pval[hosp_pair_idx, :, :] = dr_tech_shifts_samples_results_feat_p_val
                        hosp_pair_feat_pval[hosp_pair_idx, :] = mean_feat_p_vals < adjust_sign_level
                        hosp_pair_feat_tval[hosp_pair_idx, :] = mean_feat_t_vals

                    if dr == DimensionalityReduction.NoRed.value: # TODO run auc smr plots only once in dr_techniques_plot
                        feats_shifts_samples_te_auc = samples_shifts_rands_feats_hosps_te_acc[si,shift_idx,:,feature_group_idx,hosp_pair_idx,0]
                        feats_shifts_samples_te_smr = samples_shifts_rands_feats_hosps_te_acc[si,shift_idx,:,feature_group_idx,hosp_pair_idx,1]

                        mean_te_auc = np.mean(feats_shifts_samples_te_auc)
                        std_te_auc = np.std(feats_shifts_samples_te_auc)
                        mean_te_smr = np.mean(feats_shifts_samples_te_smr)
                        std_te_smr = np.std(feats_shifts_samples_te_smr)

                        hosp_pair_auc[hosp_train_idx, hosp_test_idx] = mean_te_auc
                        hosp_pair_smr[hosp_train_idx, hosp_test_idx] = mean_te_smr

                hosp_avg_pval = hosp_pair_pval.mean(axis=1)
                hosp_pair_pval_triu = np.triu(np.ones_like(hosp_pair_pval, dtype=np.bool))
                np.fill_diagonal(hosp_pair_pval_triu, False)
                hosp_pair_pval = pd.DataFrame(hosp_pair_pval, columns=HospitalIDs, index=HospitalIDs)
                hosp_pair_pval.to_csv("%s/%s_%s_%s_p_val_df.csv" % (feats_path, DimensionalityReduction(dr).name, shift, sample), index=True)
                # cmap_binary = sns.cubehelix_palette(2, hue=0.05, rot=0, light=0.9, dark=0)
                # fig = sns.heatmap(hosp_pair_pval, linewidths=0.5, cmap=ListedColormap(cmap_binary))
                fig = sns.heatmap(hosp_pair_pval, mask=hosp_pair_pval_triu, linewidths=0.5, cmap=ListedColormap(cmap_binary))
                colorbar = fig.collections[0].colorbar
                colorbar.set_ticks([0.25, 0.75])
                colorbar.set_ticklabels(['No Shift', 'Shift'])
                plt.xlabel('Hospital ID') # Target
                plt.ylabel('Hospital ID') # Source
                plt.savefig("%s/%s_%s_%s_p_val_hmp.pdf" % (feats_path, DimensionalityReduction(dr).name, shift, sample), bbox_inches='tight')
                plt.clf()
                
                # cmap = sns.cubehelix_palette(2, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True)
                if test_type == 'univ':
                    # hosp_pair_feat_pval = hosp_pair_feat_pval.min(axis=0) # Bonferroni correction by taking min across hospital pairs
                    # hosp_pair_feat_avg_pval = hosp_pair_feat_pval.mean(axis=1) < adjust_sign_level # mean across random runs
                    hosp_pair_feat_avg_pval = hosp_pair_feat_pval.mean(axis=0)
                    feature_set_escaped = [i.replace('_', '\_') for i in feature_set]
                    hosp_pair_feat_avg_pval = pd.DataFrame(hosp_pair_feat_avg_pval, index=feature_set_escaped)
                    hosp_pair_feat_avg_pval.columns=["Features"]
                    hosp_pair_feat_avg_pval.to_csv("%s/%s_%s_%s_feat_avg_pval_df.csv" % (feats_path, DimensionalityReduction(dr).name, shift, sample), index=True)
                    plt.figure(figsize=(8, 6))
                    fig = sns.heatmap(hosp_pair_feat_avg_pval, linewidths=0.5, cmap=cmap, square=True)
                    plt.savefig("%s/%s_%s_%s_feat_avg_pval_hmp.pdf" % (feats_path, DimensionalityReduction(dr).name, shift, sample), bbox_inches='tight')
                    plt.clf()

                    hosp_pair_feat_avg_tval = hosp_pair_feat_tval.mean(axis=0)
                    hosp_pair_feat_avg_tval = pd.DataFrame(hosp_pair_feat_avg_tval, index=feature_set_escaped)
                    hosp_pair_feat_avg_tval.columns=["Features"]
                    hosp_pair_feat_avg_tval.to_csv("%s/%s_%s_%s_feat_avg_tval_df.csv" % (feats_path, DimensionalityReduction(dr).name, shift, sample), index=True)
                    plt.figure(figsize=(8, 6))
                    fig = sns.heatmap(hosp_pair_feat_avg_tval, linewidths=0.5, cmap=cmap, square=True)
                    plt.savefig("%s/%s_%s_%s_feat_avg_tval_hmp.pdf" % (feats_path, DimensionalityReduction(dr).name, shift, sample), bbox_inches='tight')
                    plt.clf()

                # Minimum of the pairwise average tval in subsets of 5 hospitals
                MAX_NUM_SUBSET = 5
                HospitalIDs_ = np.array(HospitalIDs)
                for num_subset in range(1, MAX_NUM_SUBSET+1):
                    avg_tval_subset = []
                    for subs in combinations(range(len(HospitalIDs_)), num_subset):
                        avg_tval_subset.append((subs, hosp_pair_tval[np.ix_(subs,subs)].mean()))
                    avg_tval_subset_sorted = sorted(avg_tval_subset, key=lambda x: x[1])
                    avg_tval_subset_sorted = [(HospitalIDs_[np.array(subs)],mmd) for subs,mmd in avg_tval_subset_sorted]
                    avg_tval_subset_sorted = pd.DataFrame(avg_tval_subset_sorted, columns=['HospitalIDs','average_MMD'])
                    avg_tval_subset_sorted.to_csv("%s/%s_%s_%s_%s_t_val_min_subset.csv" % (feats_path, DimensionalityReduction(dr).name, shift, sample, num_subset), index=False)

                hosp_avg_tval = hosp_pair_tval.mean(axis=1)
                hosp_pair_tval_triu = np.triu(np.ones_like(hosp_pair_tval, dtype=np.bool))
                np.fill_diagonal(hosp_pair_tval_triu, False)
                hosp_pair_tval = pd.DataFrame(hosp_pair_tval, columns=HospitalIDs, index=HospitalIDs)
                hosp_pair_tval.to_csv("%s/%s_%s_%s_t_val_df.csv" % (feats_path, DimensionalityReduction(dr).name, shift, sample), index=True)
                # cmap = sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True)
                # fig = sns.heatmap(hosp_pair_tval, linewidths=0.5, cmap=cmap)
                fig = sns.heatmap(hosp_pair_tval, mask=hosp_pair_tval_triu, linewidths=0.5, cmap=cmap)
                plt.xlabel('Hospital ID') # Target
                plt.ylabel('Hospital ID') # Source
                plt.savefig("%s/%s_%s_%s_t_val_hmp.pdf" % (feats_path, DimensionalityReduction(dr).name, shift, sample), bbox_inches='tight')
                plt.clf()

                hosp_all_pairs_tval = pd.melt(hosp_pair_tval.reset_index(), id_vars='index')
                hosp_all_pairs_tval.columns = ['Source','Target','MMD']

                if dr == DimensionalityReduction.NoRed.value: # TODO run auc smr plots only once in dr_techniques_plot
                    hosp_avg_auc = hosp_pair_auc.mean(axis=1)
                    hosp_min_auc = hosp_pair_auc.min(axis=1)
                    hosp_pair_auc = pd.DataFrame(hosp_pair_auc, columns=HospitalIDs, index=HospitalIDs)
                    hosp_pair_auc.to_csv("%s/%s_%s_%s_te_val_diff_auc_df.csv" % (feats_path, DimensionalityReduction(dr).name, shift, sample), index=True)
                    # cmap = sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True)
                    fig = sns.heatmap(hosp_pair_auc, linewidths=0.5, cmap=cmap)
                    plt.xlabel('Target Hospital ID')
                    plt.ylabel('Source Hospital ID')
                    plt.savefig("%s/%s_%s_%s_te_val_diff_auc_hmp.pdf" % (feats_path, DimensionalityReduction(dr).name, shift, sample), bbox_inches='tight')
                    plt.clf()

                    hosp_all_pairs_auc = pd.melt(hosp_pair_auc.reset_index(), id_vars='index')
                    hosp_all_pairs_auc.columns = ['Source','Target','AUC']

                    hosp_avg_smr = hosp_pair_smr.mean(axis=1)
                    hosp_pair_smr = pd.DataFrame(hosp_pair_smr, columns=HospitalIDs, index=HospitalIDs)
                    hosp_pair_smr.to_csv("%s/%s_%s_%s_te_val_diff_smr_df.csv" % (feats_path, DimensionalityReduction(dr).name, shift, sample), index=True)
                    # cmap = sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True)
                    fig = sns.heatmap(hosp_pair_smr, linewidths=0.5, cmap=cmap)
                    plt.xlabel('Target Hospital ID')
                    plt.ylabel('Source Hospital ID')
                    plt.savefig("%s/%s_%s_%s_te_val_diff_smr_hmp.pdf" % (feats_path, DimensionalityReduction(dr).name, shift, sample), bbox_inches='tight')
                    plt.clf()

                    hosp_all_pairs_smr = pd.melt(hosp_pair_smr.reset_index(), id_vars='index')
                    hosp_all_pairs_smr.columns = ['Source','Target','SMR']

                    # Scatter plot
                    h_stats = hosp_all_pairs_tval.merge(hosp_all_pairs_auc, how='left',
                                                    left_on=['Source','Target'], right_on = ['Source','Target'])\
                                                .merge(hosp_all_pairs_smr, how='left',
                                                    left_on=['Source','Target'], right_on = ['Source','Target'])
                    h_stats = h_stats[h_stats.Source!=h_stats.Target]
                    
                    fig = sns.regplot(data=h_stats, x='MMD', y='AUC', scatter_kws={"s": 80, 'alpha':0.6}, truncate=False)
                    corr_coef, pval_corr_coef = stats.pearsonr(h_stats['MMD'], h_stats['AUC'])
                    textstr = '\n'.join((
                        r'Pearson corr.=%.4f' % (corr_coef, ),
                        r'p-val=%.4f' % (pval_corr_coef, )))
                    # these are matplotlib.patch.Patch properties
                    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
                    # place a text box in upper left in axes coords
                    fig.text(0.5, 0.95, textstr, transform=fig.transAxes, fontsize=14,
                            verticalalignment='top', bbox=props)
                    plt.xlabel('$MMD^2$')
                    plt.ylabel('Test AUC - Val AUC')
                    plt.savefig("%s/%s_%s_%s_mmd_te_val_diff_auc_scatter.pdf" % (feats_path, DimensionalityReduction(dr).name, shift, sample), bbox_inches='tight')
                    plt.clf()
                    
                    fig = sns.regplot(data=h_stats, x='MMD', y='SMR', scatter_kws={"s": 80, 'alpha':0.6}, truncate=False)
                    corr_coef, pval_corr_coef = stats.pearsonr(h_stats['MMD'], h_stats['SMR'])
                    textstr = '\n'.join((
                        r'Pearson corr.=%.4f' % (corr_coef, ),
                        r'p-val=%.4f' % (pval_corr_coef, )))
                    fig.text(0.5, 0.95, textstr, transform=fig.transAxes, fontsize=14,
                            verticalalignment='top', bbox=props)
                    plt.xlabel('$MMD^2$')
                    plt.ylabel('Test SMR - Val SMR')
                    plt.savefig("%s/%s_%s_%s_mmd_te_val_diff_smr_scatter.pdf" % (feats_path, DimensionalityReduction(dr).name, shift, sample), bbox_inches='tight')
                    plt.clf()

                    # h_stats = pd.DataFrame(data=np.concatenate(\
                    #     [hosp_avg_tval[:,np.newaxis], hosp_avg_auc[:,np.newaxis], hosp_min_auc[:,np.newaxis],\
                    #         hosp_avg_smr[:,np.newaxis]],axis=1),\
                    #     columns=['MMD','AUC','AUC_min','SMR'],\
                    #     index=HospitalIDs)
                                      
                    # fig = sns.scatterplot(data=h_stats, x='MMD', y='AUC', s=100, alpha=0.6)
                    # plt.xlabel('$MMD^2$, average across hospital pairs')
                    # plt.ylabel('AUC')
                    # plt.savefig("%s/%s_%s_%s_avgmmd_avgacc_scatter.pdf" % (feats_path, DimensionalityReduction(dr).name, shift, sample), bbox_inches='tight')
                    # plt.clf()

                    # fig = sns.scatterplot(data=h_stats, x='MMD', y='AUC_min', s=100, alpha=0.6)
                    # plt.xlabel('$MMD^2$, min across hospital pairs')
                    # plt.ylabel('AUC')
                    # plt.savefig("%s/%s_%s_%s_avgmmd_minacc_scatter.pdf" % (feats_path, DimensionalityReduction(dr).name, shift, sample), bbox_inches='tight')
                    # plt.clf()