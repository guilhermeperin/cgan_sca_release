import numpy as np
import os

results_root_path = "C:/Users/Sengim/Datasets/paper_9_gan_results/random_search_19_06_2023_13"

totals = {"hi":1}
successes = {"hi":1}
hp_to_check = "conv"
test_mlp = False

for subdir in os.listdir(results_root_path):
    folder_path = results_root_path + "/" + subdir
    if not os.path.isfile(folder_path + "/ge_vector.npz"):
        continue
    temp = np.load(folder_path + "/hp.npz", allow_pickle=True)['hp_g']
    value = temp[()][hp_to_check]
    if test_mlp and temp[()]['conv']:
        continue
    if (hp_to_check.__contains__("conv_") or hp_to_check.__contains__("filter")) and not temp[()]['conv']:
        continue
    success = np.load(folder_path + "/ge_vector.npz")["ge_vector_fake"][-1] <= 2
    if value in totals:
        totals[value] = totals[value] + 1
    else:
        totals[value] = 1
        successes[value] = 0
    if value in successes and success:
        successes[value] = successes[value] + 1

for key in totals:
    if key == "hi":
        continue
    print(f"For value {key} the success rate was {successes[key]}/{totals[key]}")