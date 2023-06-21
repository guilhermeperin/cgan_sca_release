import numpy as np
import os
import matplotlib.pyplot as plt

results_root_path = "C:/Users/Sengim/Datasets/paper_9_gan_results/random_search_19_06_2023_13"

totals = {"hi":1}
successes = {"hi":1}
hp_to_check = "lin_activation"

for subdir in os.listdir(results_root_path):
    folder_path = results_root_path + "/" + subdir
    if not os.path.isfile(folder_path + "/ge_vector.npz"):
        continue
    temp = np.load(folder_path + "/hp.npz", allow_pickle=True)['hp_g']
    nt = np.load(folder_path + "/metrics.npz", allow_pickle=True)['nt_fake'][0]
    print(temp, nt)
    ge = np.load(folder_path + "/max_snr_shares.npz")
    for item in ge:
        plt.plot(ge[item], label=item)
    plt.legend()
    plt.show()
