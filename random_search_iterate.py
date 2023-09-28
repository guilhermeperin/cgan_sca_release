import numpy as np
import os
import matplotlib.pyplot as plt

results_root_paths = ["C:/Users/Sengim/Datasets/paper_9_gan_results/random_search_19_06_2023_13", "C:/Users/Sengim/Datasets/paper_9_gan_results/random_search_21_06_2023_06"]


totals = {"hi":1}
successes = {"hi":1}
hp_to_check = "conv"
for results_root_path in results_root_paths:
    for subdir in os.listdir(results_root_path):
        folder_path = results_root_path + "/" + subdir
        if not os.path.isfile(folder_path + "/ge_vector.npz"):
            continue
        temp = np.load(folder_path + "/hp.npz", allow_pickle=True)['hp_g']
        temp2 = np.load(folder_path + "/hp.npz", allow_pickle=True)['hp_d']
        nt = np.load(folder_path + "/metrics.npz", allow_pickle=True)['nt_fake'][0]
        print(temp2, temp, nt)
        if nt ==2000:
            continue
        ge = np.load(folder_path + "/max_snr_shares.npz")
        for item in ge:
            if item == 'max_snr_share_2':
                label = "Maximum SNR value share 2"
            else:
                label = "Maximum SNR value share 1"
            plt.plot(ge[item], label=label)
        plt.xlabel("Epochs")
        plt.ylabel("SNR")
        plt.legend()
        plt.show()
