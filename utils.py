import numpy as np
from datetime import datetime
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from src.datasets.paths import *
from src.datasets.load_ascadr import *
from src.datasets.load_ascadf import *
from src.datasets.load_dpav42 import *
from src.datasets.load_eshard import *
from src.datasets.load_chesctf import *
from os.path import exists


def snr_fast(x, y):
    ns = x.shape[1]
    unique = np.unique(y)
    means = np.zeros((len(unique), ns))
    variances = np.zeros((len(unique), ns))

    for i, u in enumerate(unique):
        new_x = x[np.argwhere(y == int(u))]
        means[i] = np.mean(new_x, axis=0)
        variances[i] = np.var(new_x, axis=0)
    return np.var(means, axis=0) / np.mean(variances, axis=0)

def get_features_bit(dataset, target_byte: int, n_poi=100):
    poi = np.zeros(n_poi, dtype=np.int32)
    per_bit =  n_poi//16
    for i in range(8):
        poi[i*per_bit: (i+1)*per_bit] = get_features_bit_per(np.array(dataset.x_profiling[:20000], dtype=np.int16), np.asarray(dataset.share1_profiling[target_byte, :20000], dtype=np.uint8), i,per_bit)
        poi[n_poi//2 + i*per_bit:n_poi//2 + (i+1)*per_bit] = get_features_bit_per(np.array(dataset.x_profiling[:20000], dtype=np.int16), np.asarray(dataset.share2_profiling[target_byte, :20000], dtype=np.uint8), i,per_bit)
    return dataset.x_profiling[:, poi], dataset.x_attack[:, poi]


def get_features(dataset, target_byte: int, n_poi=100):
    snr_val_share_1 = snr_fast(np.array(dataset.x_profiling[:min(20000, dataset.x_profiling.shape[0])], dtype=np.int16), np.asarray(dataset.share1_profiling[target_byte, :min(20000, dataset.x_profiling.shape[0])]))
    snr_val_share_2 = snr_fast(np.array(dataset.x_profiling[:min(20000, dataset.x_profiling.shape[0])], dtype=np.int16), np.asarray(dataset.share2_profiling[target_byte, :min(20000, dataset.x_profiling.shape[0])]))
    snr_val_share_1[np.isnan(snr_val_share_1)] = 0
    snr_val_share_2[np.isnan(snr_val_share_2)] = 0
    
    ind_snr_masks_poi_sm = np.argsort(snr_val_share_1)[::-1][:int(n_poi / 2)]
    sorted_poi_masks_sm = np.argsort(snr_val_share_1)[::-1]
    ind_snr_masks_poi_sm_sorted = np.sort(ind_snr_masks_poi_sm)

    ind_snr_masks_poi_r2 = np.argsort(snr_val_share_2)[::-1][:int(n_poi / 2)]
    sorted_poi_masks_r2 = np.argsort(snr_val_share_2)[::-1]
    ind_snr_masks_poi_r2_sorted = np.sort(ind_snr_masks_poi_r2)

    poi_profiling = np.concatenate((ind_snr_masks_poi_sm_sorted, ind_snr_masks_poi_r2_sorted), axis=0)

    return dataset.x_profiling[:, poi_profiling], dataset.x_attack[:, poi_profiling]

def get_lda_features(dataset, target_byte: int, n_components=10):
    x_prof, x_att = get_features(dataset, target_byte, n_poi=200)
    lda_s1 = LinearDiscriminantAnalysis(n_components=n_components//2)
    lda_s1.fit(x_prof[:5000, :100], np.asarray(dataset.share1_profiling[target_byte, :5000]))
    lda_s2 = LinearDiscriminantAnalysis(n_components=n_components//2)
    lda_s2.fit(x_prof[:5000, 100:], np.asarray(dataset.share2_profiling[target_byte, :5000]))
    s1_prof = lda_s1.transform(x_prof[:, :100])
    s1_att = lda_s1.transform(x_att[:, :100])
    s2_prof = lda_s2.transform(x_prof[:, 100:])
    s2_att = lda_s2.transform(x_att[:, 100:])
    return np.append(s1_prof, s2_prof, axis=1), np.append(s1_att, s2_att, axis=1)

def get_pca_features(dataset, target_byte: int, n_components=10):
    x_prof, x_att = get_features(dataset, target_byte, n_poi=200)

    pca = PCA(n_components=n_components//2)
    pca.fit(x_prof[:20000, :100])
    s1_prof = pca.transform(x_prof[:, :100] )
    s1_att = pca.transform(x_att[:, :100] )
    pca = PCA(n_components=n_components//2)
    pca.fit(x_prof[:20000, 100:])
    s2_prof = pca.transform(x_prof[:, 100:] )
    s2_att = pca.transform(x_att[:, 100:] )
    return np.append(s1_prof, s2_prof, axis=1), np.append(s1_att, s2_att, axis=1)




def create_directory_results(args, path):
    now = datetime.now()
    now_str = f"{now.strftime('%d_%m_%Y_%H_%M_%S')}_{np.random.randint(1000000, 10000000)}"
    dir_results = f"{path}/{args['dataset_reference']}_vs_{args['dataset_target']}_{now_str}"
    if not os.path.exists(dir_results):
        os.mkdir(dir_results)
    return dir_results


def get_features_bit_per(x, y, bit, points):
    temp = snr_fast(x, (y>>(7-bit)) & 1)
    ind_snr_masks_poi_sm = np.argsort(temp)[::-1][:points]
    ind_snr_masks_poi_sm_sorted = np.sort(ind_snr_masks_poi_sm)
    return ind_snr_masks_poi_sm_sorted

def load_dataset(identifier: str, path: str, target_byte: int, traces_dim: int, leakage_model="ID", num_features=-1):
    
    dataset_file = get_dataset_filepath(path, identifier, traces_dim, leakage_model=leakage_model)
    snr_shortcut = f'{path}/paper_9_gan_features/selected_{num_features}_features_snr_{identifier}_{traces_dim}.h5'
    if num_features > 0 and exists(snr_shortcut):
        dataset_file = snr_shortcut
        traces_dim = num_features
        
    if identifier == "ascad-variable":
        dataset = ReadASCADr(200000, 0, 10000, target_byte, leakage_model,
                                                dataset_file,
                                                number_of_samples=traces_dim)
    if identifier == "ASCAD":
        dataset = ReadASCADf(50000, 0, 10000, target_byte, leakage_model,
                                                dataset_file,
                                                number_of_samples=traces_dim)
    if identifier == "dpa_v42":
        dataset = ReadDPAV42(70000, 0, 5000, target_byte, leakage_model,
                                                dataset_file,
                                                number_of_samples=traces_dim)
    if identifier == "ches_ctf":
        dataset = ReadCHESCTF(45000, 0, 5000, target_byte, leakage_model,
                                                         dataset_file,
                                                         number_of_samples=traces_dim)
    return dataset

def scale_dataset(prof_set, attack_set, scaler):
        prof_new = scaler.fit_transform(prof_set)
        if attack_set is not None:
            attack_new = scaler.transform(attack_set)
        else:
            attack_new = None
        return prof_new, attack_new