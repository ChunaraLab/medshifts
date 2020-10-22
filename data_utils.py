# -------------------------------------------------
# IMPORTS
# -------------------------------------------------

import numpy as np
import scipy.io
from math import ceil
from sklearn.impute import SimpleImputer

from keras.datasets import mnist, cifar10, cifar100, boston_housing, fashion_mnist
from keras.preprocessing.image import ImageDataGenerator

# To use the experimental IterativeImputer, we need to explicitly ask for it:
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import (
    SimpleImputer, KNNImputer, IterativeImputer, MissingIndicator)

import pandas as pd

# -------------------------------------------------
# DATA UTILS
# -------------------------------------------------

# Define proportion of examples in train set, 1-TRAIN_PROP in validation
TRAIN_PROP = 0.90

# Define feature groups
# 'ventdays'
FeatureGroups_eicu = {
    'outcome': 'death',
    'hospital': ['hospitalid', 'hosp_los'],
    'labs': ['albumin_first_early',
       'bands_first_early', 'bicarbonate_first_early', 'bilirubin_first_early',
       'bun_first_early', 'calcium_first_early', 'creatinine_first_early',
       'hematocrit_first_early',
       'hemoglobin_first_early', 'inr_first_early', 'lactate_first_early',
       'platelet_first_early', 'potassium_first_early', 'sodium_first_early',
       'wbc_first_early',
       'albumin_last_early', 'bands_last_early',
       'bicarbonate_last_early', 'bilirubin_last_early', 'bun_last_early',
       'calcium_last_early', 'creatinine_last_early',
       'hematocrit_last_early', 'hemoglobin_last_early', 'inr_last_early',
       'lactate_last_early', 'platelet_last_early', 'potassium_last_early',
       'sodium_last_early', 'wbc_last_early'],
    'vitals': ['heartrate_first',
       'sysbp_first', 'diasbp_first', 'meanbp_first', 'resprate_first',
       'tempc_first', 'spo2_first', 'gcs_first', 'bg_pao2_first_early',
       'bg_paco2_first_early', 'bg_pao2fio2ratio_first_early',
       'bg_ph_first_early', 'bg_baseexcess_first_early', 'glucose_first_early',
       'heartrate_last', 'sysbp_last', 'diasbp_last',
       'meanbp_last', 'resprate_last', 'tempc_last', 'spo2_last', 'gcs_last',
       'bg_pao2_last_early', 'bg_paco2_last_early',
       'bg_pao2fio2ratio_last_early', 'bg_ph_last_early',
       'bg_baseexcess_last_early', 'glucose_last_early',
       'heartrate_min', 'sysbp_min',
       'diasbp_min', 'meanbp_min', 'resprate_min', 'tempc_min', 'spo2_min', 'gcs_min',
       'heartrate_max', 'sysbp_max', 'diasbp_max', 'meanbp_max',
       'resprate_max', 'tempc_max', 'spo2_max', 'gcs_max',
       'urineoutput_sum'],
    'demo': ['is_female', 'age', 'race_black', 'race_hispanic', 'race_asian', 'race_other'],
    'others': ['electivesurgery'],
    'saps2diff': ['heartrate',
       'sysbp',
       'sodium',
       'potassium',
       'wbc'],
    'saps2': ['heartrate',
       'sysbp',
       'temp',
       'bg_pao2fio2ratio',
       'bun',
       'urineoutput',
       'sodium',
       'potassium',
       'bicarbonate',
       'bilirubin',
       'wbc',
       'gcs',
       'age',
       'electivesurgery'],
    'saps2labs': ['bun',
        'sodium',
        'potassium',
        'bicarbonate',
        'bilirubin',
        'wbc'
    ],
    'saps2vitals': ['heartrate',
        'sysbp',
        'temp',
        'bg_pao2fio2ratio',
        'urineoutput',
        'gcs'
    ]
}

# 'demographic': ['age', 'bmi', 'elective_surgery', 'ethnicity', 'gender',
#                     'height', 'hospital_admit_source', 'icu_admit_source', 'icu_id',
#                     'icu_stay_type', 'icu_type', 'pre_icu_los_days', 'readmission_status', 'weight']
FeatureGroups_gossis = {
    'outcome': 'hospital_death',
    'demographic': ['age', 'bmi', 'elective_surgery', 'height', 'pre_icu_los_days',
                    'readmission_status', 'weight', 'ethnicity_African American',
                    'ethnicity_Asian', 'ethnicity_Caucasian', 'ethnicity_Hispanic',
                    'ethnicity_Native American', 'ethnicity_Other/Unknown', 'gender_F',
                    'gender_M'],
    'APACHE_covariate': ['albumin_apache', 'apache_2_diagnosis', 'apache_3j_diagnosis', 'apache_post_operative',
                        'arf_apache', 'bilirubin_apache', 'bun_apache', 'creatinine_apache', 'fio2_apache',
                        'gcs_eyes_apache', 'gcs_motor_apache', 'gcs_unable_apache', 'gcs_verbal_apache',
                        'glucose_apache', 'heart_rate_apache', 'hematocrit_apache', 'intubated_apache',
                        'map_apache', 'paco2_apache', 'paco2_for_ph_apache', 'pao2_apache', 'ph_apache', 'resprate_apache', 'sodium_apache',
                        'temp_apache', 'urineoutput_apache', 'ventilated_apache', 'wbc_apache'],
    'vitals': ['d1_diasbp_invasive_max', 'd1_diasbp_invasive_min', 'd1_diasbp_max', 'd1_diasbp_min',
                'd1_diasbp_noninvasive_max', 'd1_diasbp_noninvasive_min', 'd1_heartrate_max', 'd1_heartrate_min',
                'd1_mbp_invasive_max', 'd1_mbp_invasive_min', 'd1_mbp_max', 'd1_mbp_min', 'd1_mbp_noninvasive_max',
                'd1_mbp_noninvasive_min', 'd1_resprate_max', 'd1_resprate_min', 'd1_spo2_max', 'd1_spo2_min',
                'd1_sysbp_invasive_max', 'd1_sysbp_invasive_min', 'd1_sysbp_max', 'd1_sysbp_min', 'd1_sysbp_noninvasive_max',
                'd1_sysbp_noninvasive_min', 'd1_temp_max', 'd1_temp_min', 'h1_diasbp_invasive_max',
                'h1_diasbp_invasive_min', 'h1_diasbp_max', 'h1_diasbp_min', 'h1_diasbp_noninvasive_max',
                'h1_diasbp_noninvasive_min', 'h1_heartrate_max', 'h1_heartrate_min', 'h1_mbp_invasive_max',
                'h1_mbp_invasive_min', 'h1_mbp_max', 'h1_mbp_min', 'h1_mbp_noninvasive_max', 'h1_mbp_noninvasive_min',
                'h1_resprate_max', 'h1_resprate_min', 'h1_spo2_max', 'h1_spo2_min', 'h1_sysbp_invasive_max',
                'h1_sysbp_invasive_min', 'h1_sysbp_max', 'h1_sysbp_min', 'h1_sysbp_noninvasive_max',
                'h1_sysbp_noninvasive_min', 'h1_temp_max', 'h1_temp_min'],
    'labs': ['d1_albumin_max', 'd1_albumin_min', 'd1_bilirubin_max', 'd1_bilirubin_min',
            'd1_bun_max', 'd1_bun_min', 'd1_calcium_max', 'd1_calcium_min', 'd1_creatinine_max', 'd1_creatinine_min',
            'd1_glucose_max', 'd1_glucose_min', 'd1_hco3_max', 'd1_hco3_min',
            'd1_hemaglobin_max', 'd1_hemaglobin_min', 'd1_hematocrit_max', 'd1_hematocrit_min',
            'd1_inr_max', 'd1_inr_min', 'd1_lactate_max', 'd1_lactate_min', 'd1_platelets_max',
            'd1_platelets_min', 'd1_potassium_max', 'd1_potassium_min', 'd1_sodium_max', 'd1_sodium_min',
            'd1_wbc_max', 'd1_wbc_min', 'h1_albumin_max', 'h1_albumin_min', 'h1_bilirubin_max', 'h1_bilirubin_min',
            'h1_bun_max', 'h1_bun_min', 'h1_calcium_max', 'h1_calcium_min', 'h1_creatinine_max', 'h1_creatinine_min',
            'h1_glucose_max', 'h1_glucose_min', 'h1_hco3_max', 'h1_hco3_min', 'h1_hemaglobin_max', 'h1_hemaglobin_min',
            'h1_hematocrit_max', 'h1_hematocrit_min', 'h1_inr_max', 'h1_inr_min', 'h1_lactate_max', 'h1_lactate_min',
            'h1_platelets_max', 'h1_platelets_min', 'h1_potassium_max', 'h1_potassium_min',
            'h1_sodium_max', 'h1_sodium_min', 'h1_wbc_max', 'h1_wbc_min'],
    'labs_blood_gas': ['d1_arterial_pco2_max', 'd1_arterial_pco2_min', 'd1_arterial_ph_max',
                        'd1_arterial_ph_min', 'd1_arterial_po2_max', 'd1_arterial_po2_min',
                        'd1_pao2fio2ratio_max', 'd1_pao2fio2ratio_min', 'h1_arterial_pco2_max', 'h1_arterial_pco2_min',
                        'h1_arterial_ph_max', 'h1_arterial_ph_min', 'h1_arterial_po2_max', 'h1_arterial_po2_min',
                        'h1_pao2fio2ratio_max', 'h1_pao2fio2ratio_min'],
    'APACHE_prediction': ['apache_4a_hospital_death_prob', 'apache_4a_icu_death_prob'],
    'APACHE_comorbidity': ['aids', 'cirrhosis', 'diabetes_mellitus', 'hepatic_failure', 'immunosuppression',
                            'leukemia', 'lymphoma', 'solid_tumor_with_metastasis'],
    'APACHE_grouping': ['apache_3j_bodysystem', 'apache_2_bodysystem']
}

HospitalIDs_eicu = [73, 264, 338, 443, 458, 420, 252, 300, 122, 243, 188, 449, 208,
       307, 416, 413, 394, 199, 345]
HospitalIDs_gossis = [118, 19, 188, 161, 70, 196, 176, 21, 194, 174, 100, 55,
                    185, 79, 18, 157, 62, 39, 112, 76]

HospitalGroups_eicu = ['X_eicu_day1_saps2_n500_teach_midw.csv',
                        'X_eicu_day1_saps2_n500_teach_s.csv',
                        'X_eicu_day1_saps2_n500_noteach_midw.csv',
                        'X_eicu_day1_saps2_n500_noteach_s.csv',
                        'X_eicu_day1_saps2_nl249_noteach_midw.csv',
                        'X_eicu_day1_saps2_nl249_noteach_s.csv']
HospitalGroupsColnames_eicu = ['nbed500,teach,midwest',
                        'nbed500,teach,south',
                        'nbed500,noteach,midwest',
                        'nbed500,noteach,south',
                        'nbedl250,noteach,midwest',
                        'nbedl250,noteach,south']

def __unison_shuffled_copies(a, b, c):
    assert len(a) == len(b)
    assert len(a) == len(c)
    p = np.random.permutation(len(a))
    return a[p], b[p], c[p]


def normalize_datapoints(x, factor):
    x = x.astype('float32') / factor
    return x


def random_shuffle(x, y, z):
    x, y, z = __unison_shuffled_copies(x, y, z)
    return x, y, z


def random_shuffle_and_split(x_train, y_train, sens_train, x_test, y_test, sens_test, split_index):
    x = np.append(x_train, x_test, axis=0)
    y = np.append(y_train, y_test, axis=0)
    sens = np.append(sens_train, sens_test, axis=0)

    x, y, sens = __unison_shuffled_copies(x, y, sens)

    x_train = x[:split_index, :]
    x_test = x[split_index:, :]
    y_train = y[:split_index]
    y_test = y[split_index:]
    sens_train = sens[:split_index] # assume 1d array for 1 sensitive feature
    sens_test = sens[split_index:]

    return (x_train, y_train, sens_train), (x_test, y_test, sens_test)

def import_hosp_dataset(dataset):
    df = None
    external_dataset_path = './datasets/'
    if dataset == 'eicu':
        '''
        From https://github.com/alistairewj/icu-model-transfer/blob/master/evaluate-model.ipynb
        '''
        df = pd.read_csv(external_dataset_path + 'X_eicu_day1.csv.gz', sep=',', index_col=0)
    return df

def load_hosp_dataset(dataset, df, target, features, hosp_train, hosp_test, use_group, shuffle=False):
    '''
    :param hosp_train, hosp_test: hospital IDs to include in train and test set
    :param min_trans: minimum transactions per hospital for inclusion
    '''
    x_train, y_train, x_test, y_test = None, None, None, None
    external_dataset_path = './datasets/'
    nb_classes = 2
    if dataset == 'eicu':
        data_filename = 'X_eicu_day1_saps2.csv'
        index_col = 0
        hospitalid_var = 'hospitalid'
        var_other = ['hospitalid', 'death', 'hosp_los', 'ventdays']
        sensitive_feature = 'is_female'
    elif dataset == 'gossis':
        data_filename = 'training_v2_top15hosp_dummy_gossis.csv'
        index_col = None
        hospitalid_var = 'hospital_id'
        var_other = ['hospital_id', 'hospital_death', 'encounter_id', 'patient_id']
        sensitive_feature = 'gender_F'

    # df_eicu = df.copy()
    '''
    From https://github.com/alistairewj/icu-model-transfer/blob/master/evaluate-model.ipynb
    '''

    if use_group:
        hosp_train = hosp_train[0]
        hosp_test = hosp_test[0]
        df_eicu_train = pd.read_csv(external_dataset_path + hosp_train, sep=',', index_col=index_col)
        df_eicu_test = pd.read_csv(external_dataset_path + hosp_test, sep=',', index_col=index_col)
        # print(df_eicu_train.head())
        # print(df_eicu_test.head())

    else:
        df_eicu = pd.read_csv(external_dataset_path + data_filename, sep=',', index_col=index_col) # TODO move filename to main file
        # hosp_to_keep = df_eicu[hospitalid_var].value_counts()
        # hosp_to_keep = hosp_to_keep[hosp_to_keep>=min_trans].index.values
        # print('Retaining {} of {} hospitals with at least 100 patients.'.format(
        #     len(hosp_to_keep), df_eicu[hospitalid_var].nunique()))
        df_eicu_train = df_eicu.loc[df_eicu[hospitalid_var].isin(np.array(hosp_train)), :]
        df_eicu_test = df_eicu.loc[df_eicu[hospitalid_var].isin(np.array(hosp_test)), :]

    # Extract required features. Remove target and other vars
    y_train = df_eicu_train[target].values
    x_train = df_eicu_train.drop(var_other,axis=1)
    x_train = df_eicu_train[features].values
    sens_train = df_eicu_train[sensitive_feature].values

    y_test = df_eicu_test[target].values
    x_test = df_eicu_test.drop(var_other,axis=1)
    x_test = df_eicu_test[features].values
    sens_test = df_eicu_test[sensitive_feature].values

    # Remove features with all nan from BOTH train and test
    all_nan_train = np.all(np.isnan(x_train), axis=0)
    all_nan_test = np.all(np.isnan(x_test), axis=0)
    all_nan = np.logical_or(all_nan_train, all_nan_test)
    x_train = x_train[:, ~all_nan]
    x_test = x_test[:, ~all_nan]
    # print('Features removed', np.array(features)[all_nan], np.sum(all_nan))

    # Impute NaN by mean of column
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    x_train = imp.fit(x_train).transform(x_train)

    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    x_test = imp.fit(x_test).transform(x_test)

    # # Split into train and test
    # idx = np.random.permutation(x_eicu.shape[0])
    # split_idx = int(0.8*x_eicu.shape[0])
    # train_idx, test_idx = idx[:split_idx], idx[split_idx:]
    # x_train, y_train = x_eicu[train_idx,:], y_eicu[train_idx]
    # x_test, y_test = x_eicu[test_idx,:], y_eicu[test_idx]

    if shuffle:
        x_train = np.append(x_train, sens_train, axis=1) # add sensitive feature at end and remove after permuting
        x_test = np.append(x_test, sens_test, axis=1)
        (x_train, y_train), (x_test, y_test) = random_shuffle_and_split(x_train, y_train, sens_train, x_test, y_test, sens_test, len(x_train))
        sens_train = x_train[:,-1]
        sens_test = x_test[:,-1]
        x_train = x_train[:,:-1]
        x_test = x_test[:,:-1]

    # Add 3-way split
    train_size = int(len(x_train)*TRAIN_PROP)
    x_train_spl = np.split(x_train, [train_size])
    y_train_spl = np.split(y_train, [train_size])
    sens_train_spl = np.split(sens_train, [train_size])
    x_train = x_train_spl[0]
    x_val = x_train_spl[1]
    y_train = y_train_spl[0]
    y_val = y_train_spl[1]
    sens_train = sens_train_spl[0]
    sens_val = sens_train_spl[1]

    orig_dims = x_train.shape[1:]

    # Reshape to matrix form
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_val = x_val.reshape((len(x_val), np.prod(x_val.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    y_train = y_train.reshape(len(y_train))
    y_val = y_val.reshape(len(y_val))
    y_test = y_test.reshape(len(y_test))
    sens_train = sens_train.reshape(len(sens_train))
    sens_val = sens_val.reshape(len(sens_val))
    sens_test = sens_test.reshape(len(sens_test))
    # print('hosp_train, hosp_test, orig_dims, new_dims train, val, test x, y', hosp_train, hosp_test, orig_dims, x_train.shape, y_train.shape, x_val.shape, y_val.shape, x_test.shape, y_test.shape)

    return (x_train, y_train, sens_train), (x_val, y_val, sens_val), (x_test, y_test, sens_test), orig_dims, nb_classes


def import_dataset(dataset, shuffle=False):
    x_train, y_train, x_test, y_test = None, None, None, None
    external_dataset_path = './datasets/'
    nb_classes = 10
    if dataset == 'boston':
        (x_train, y_train), (x_test, y_test) = boston_housing.load_data()
    elif dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(len(x_train), 28, 28, 1)
        x_test = x_test.reshape(len(x_test), 28, 28, 1)
        x_train, y_train, x_test, y_test = x_train[:10000,], y_train[:10000], x_test[:1000,], y_test[:1000]
    elif dataset == 'mnist_adv':
        (x_train, y_train), (_, _) = mnist.load_data()
        x_test = np.loadtxt(external_dataset_path + 'mnist_X_adversarial.csv', delimiter=',')
        y_test = np.loadtxt(external_dataset_path + 'mnist_y_adversarial.csv', delimiter=',')
    elif dataset == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    elif dataset == 'cifar10_1':
        (x_train, y_train), (_, _) = cifar10.load_data()
        x_test = np.load(external_dataset_path + 'cifar10_1_v6_X.npy')
        y_test = np.load(external_dataset_path + 'cifar10_1_v6_y.npy')
        y_test = y_test.reshape((len(y_test),1))
    elif dataset == 'cifar10_adv':
        (x_train, y_train), (_, _) = cifar10.load_data()
        x_test = np.load(external_dataset_path + 'cifar10_adv_img.npy')
        y_test = np.load(external_dataset_path + 'cifar10_adv_label.npy')
        y_test = np.argmax(y_test, axis=1)
    elif dataset == 'cifar100':
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    elif dataset == 'fashion_mnist':
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x_train = x_train.reshape(len(x_train), 28, 28, 1)
        x_test = x_test.reshape(len(x_test), 28, 28, 1)
    elif dataset == 'svhn':
        train = scipy.io.loadmat(external_dataset_path + 'svhn_train.mat')
        x_train = train['X']
        x_train = np.moveaxis(x_train, -1, 0)
        y_train = train['y']
        y_train[y_train == 10] = 0
        test = scipy.io.loadmat(external_dataset_path + 'svhn_test.mat')
        x_test = test['X']
        x_test = np.moveaxis(x_test, -1, 0)
        y_test = test['y']
        y_test[y_test == 10] = 0
    elif dataset == 'stl10':
        train = scipy.io.loadmat(external_dataset_path + 'stl10_train.mat')
        x_train = train['X']
        x_train = x_train.reshape(len(x_train), 96, 96, 3)
        y_train = train['y']
        y_train[y_train == 10] = 0
        test = scipy.io.loadmat(external_dataset_path + 'stl10_test.mat')
        x_test = test['X']
        x_test = x_test.reshape(len(x_test), 96, 96, 3)
        y_test = test['y']
        y_test[y_test == 10] = 0
    elif dataset == 'mnist_usps':
        data = scipy.io.loadmat(external_dataset_path + 'MNIST_vs_USPS.mat')
        x_train = data['X_src'].T
        x_test = data['X_tar'].T
        y_train = data['Y_src']
        y_test = data['Y_tar']

        y_train[y_train == 10] = 0
        y_test[y_test == 10] = 0

        x_train = x_train.reshape((len(x_train), 16, 16, 1))
        x_test = x_test.reshape((len(x_test), 16, 16, 1))

        x_train = x_train.astype(np.float32)
        x_test = x_test.astype(np.float32)
    elif dataset == 'coil100':
        x = np.load(external_dataset_path + 'coil100_X.npy')
        y = np.load(external_dataset_path + 'coil100_y.npy')
        
        cats = 100
        nb_classes = cats
        samples_per_cat = 72
        
        x = x[:cats * samples_per_cat,:]
        y = y[:cats * samples_per_cat]
        
        img_size = 32
        img_channels = 3
        train_samples_per_cat = 72 * 2 // 3
        train_samples = train_samples_per_cat * cats
        test_samples_per_cat = samples_per_cat - train_samples_per_cat
        test_samples = test_samples_per_cat * cats
        
        x_train = np.ones((train_samples , img_size, img_size, img_channels)) * (-1)
        y_train = np.ones(train_samples) * (-1)
        x_test = np.ones((test_samples , img_size, img_size, img_channels)) * (-1)
        y_test = np.ones(test_samples) * (-1)
        
        i = 0
        j = 0
        while i < len(x):
            x_train[i:i+train_samples_per_cat] = x[j:j+train_samples_per_cat]
            y_train[i:i+train_samples_per_cat] = y[j:j+train_samples_per_cat]
            i = i + train_samples_per_cat
            j = j + samples_per_cat
            
        i = 0
        j = 0
        while i < len(x):
            x_test[i:i+test_samples_per_cat] = x[j+train_samples_per_cat:j+samples_per_cat]
            y_test[i:i+test_samples_per_cat] = y[j+train_samples_per_cat:j+samples_per_cat]
            i = i + test_samples_per_cat
            j = j + samples_per_cat

    if shuffle:
        (x_train, y_train), (x_test, y_test) = random_shuffle_and_split(x_train, y_train, x_test, y_test, len(x_train))

    if dataset == 'mnist_usps':
        x_test = x_test[:1000,:]
        y_test = y_test[:1000]    

    # Add 3-way split
    x_train_spl = np.split(x_train, [len(x_train) - len(x_test)])
    y_train_spl = np.split(y_train, [len(y_train) - len(y_test)])
    x_train = x_train_spl[0]
    x_val = x_train_spl[1]
    y_train = y_train_spl[0]
    y_val = y_train_spl[1]

    orig_dims = x_train.shape[1:]

    # Reshape to matrix form
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_val = x_val.reshape((len(x_val), np.prod(x_val.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    y_train = y_train.reshape(len(y_train))
    y_val = y_val.reshape(len(y_val))
    y_test = y_test.reshape(len(y_test))
    
    print(orig_dims)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test), orig_dims, nb_classes


# Merge clean and perturbed data based on given percentage.
def data_subset(x_clean, y_clean, x_altered, y_altered, delta=1.0):
    indices = np.random.choice(x_clean.shape[0], ceil(x_clean.shape[0] * delta), replace=False)
    indices_altered = np.random.choice(x_clean.shape[0], ceil(x_clean.shape[0] * delta), replace=False)
    x_clean[indices, :] = x_altered[indices_altered, :]
    y_clean[indices] = y_altered[indices_altered]
    return x_clean, y_clean, indices


# Perform image perturbations.
def image_generator(x, orig_dims, rot_range, width_range, height_range, shear_range, zoom_range, horizontal_flip, vertical_flip, delta=1.0):
    indices = np.random.choice(x.shape[0], ceil(x.shape[0] * delta), replace=False)
    datagen = ImageDataGenerator(rotation_range=rot_range,
                                 width_shift_range=width_range,
                                 height_shift_range=height_range,
                                 shear_range=shear_range,
                                 zoom_range=zoom_range,
                                 horizontal_flip=horizontal_flip,
                                 vertical_flip=vertical_flip,
                                 fill_mode="nearest")
    x_mod = x[indices, :]
    for idx in range(len(x_mod)):
        img_sample = x_mod[idx, :].reshape(orig_dims)
        mod_img_sample = datagen.flow(np.array([img_sample]), batch_size=1)[0]
        x_mod[idx, :] = mod_img_sample.reshape(np.prod(mod_img_sample.shape))
    x[indices, :] = x_mod
    
    return x, indices


def gaussian_noise(x, noise_amt, normalization=1.0, clip=True):
    noise = np.random.normal(0, noise_amt / normalization, (x.shape[0], x.shape[1]))
    if clip:
        return np.clip(x + noise, 0., 1.)
    else:
        return x + noise


def gaussian_noise_subset(x, noise_amt, normalization=1.0, delta_total=1.0, clip=True):
    indices = np.random.choice(x.shape[0], ceil(x.shape[0] * delta_total), replace=False)
    x_mod = x[indices, :]
    noise = np.random.normal(0, noise_amt / normalization, (x_mod.shape[0], x_mod.shape[1]))
    if clip:
        x_mod = np.clip(x_mod + noise, 0., 1.)
    else:
        x_mod = x_mod + noise
    x[indices, :] = x_mod
    return x, indices


# Remove instances of a single class.
def knockout_shift(x, y, cl, delta):
    del_indices = np.where(y == cl)[0]
    until_index = ceil(delta * len(del_indices))
    if until_index % 2 != 0:
        until_index = until_index + 1
    del_indices = del_indices[:until_index]
    x = np.delete(x, del_indices, axis=0)
    y = np.delete(y, del_indices, axis=0)
    return x, y


# Remove all classes except for one via multiple knock-out.
def only_one_shift(X, y, c):
    I = len(np.unique(y))
    i = 0
    while i < I:
        if i == c:
            i = i + 1
            continue
        X, y = knockout_shift(X, y, i, 1.0)
        i = i + 1
    return X, y


def adversarial_samples(dataset):
    x_test, y_test = None, None
    external_dataset_path = './datasets/'
    if dataset == 'mnist':
        x_test = np.load(external_dataset_path + 'mnist_X_adversarial.npy')
        y_test = np.load(external_dataset_path + 'mnist_y_adversarial.npy')
    elif dataset == 'cifar10':
        x_test = np.load(external_dataset_path + 'cifar10_X_adversarial.npy')
        y_test = np.load(external_dataset_path + 'cifar10_y_adversarial.npy')
    return x_test, y_test
