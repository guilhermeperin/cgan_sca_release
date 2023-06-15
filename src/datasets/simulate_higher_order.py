import numpy as np


def hw(input: np.uint8):
    out = 0
    temp = input
    for i in range(8):
        if temp % 2 == 1:
            out = out + 1
        temp = temp >> 1
    return out


class SimulateHigherOrder():

    def __init__(self, order, num_traces, num_attack_traces, num_informative_features, num_features) -> None:

        self.order = order
        self.num_traces = num_traces
        self.n_profiling = num_traces
        self.n_attack = num_attack_traces
        self.num_features = num_features
        self.num_informative_features = num_informative_features
        self.num_leakage_regions = 3
        if num_features // (order + 1) == num_informative_features:
            self.x_profiling, self.profiling_masks, self.profiling_shares = self.only_informative(num_traces)
            self.x_attack, self.attack_masks, self.attack_shares = self.only_informative(num_attack_traces)
        else:
            self.create_pattern(num_informative_features // self.num_leakage_regions, 20)
            indices = np.random.randint(num_features, size=self.num_leakage_regions * (self.order + 1))
            self.x_profiling, self.profiling_masks, self.profiling_shares = self.generate_traces(num_traces, indices)
            self.x_attack, self.attack_masks, self.attack_shares = self.generate_traces(num_attack_traces, indices)

        self.profiling_labels = self.profiling_masks[:, order]
        self.attack_labels = self.attack_masks[:, order]

    def generate_traces(self, num_traces, leakage_region_indices):

        masks = np.random.randint(256, size=(num_traces, self.order + 1), dtype=np.uint8)
        shares = np.zeros((num_traces, self.order + 1), dtype=np.uint8)

        shares[:, 0] = masks[:, 0]
        for i in range(1, self.order + 1):
            shares[:, i] = shares[:, i - 1] ^ masks[:, i]

        vec_hw = np.vectorize(hw)
        leakage_values = vec_hw(shares)

        traces = np.random.normal(0, 2, size=(num_traces, self.num_features))
        # How to include the actual leakage values is maybe a problem.
        # Perhaps try to put leakages of specific value in clusters, because current implementation seems unrealistic and problematic.

        for i in range(self.order + 1):
            for j in range(self.num_leakage_regions):
                traces = self.include_leakage_around_index(traces, leakage_region_indices[i * self.num_leakage_regions + j], i,
                                                           leakage_values)
        return traces, masks, shares

    def only_informative(self, num_traces):

        masks = np.random.randint(256, size=(num_traces, self.order + 1), dtype=np.uint8)
        shares = np.zeros((num_traces, self.order + 1), dtype=np.uint8)

        shares[:, 0] = masks[:, 0]
        for i in range(1, self.order + 1):
            shares[:, i] = shares[:, i - 1] ^ masks[:, i]

        vec_hw = np.vectorize(hw)
        leakage_values = vec_hw(shares)
        traces = np.random.normal(0, 5, size=(num_traces, self.num_features))

        for i in range(self.order + 1):
            for j in range(self.num_informative_features):
                traces[:, i * self.num_informative_features + j] += leakage_values[:, i] * 10

        return traces, masks, shares

    def include_leakage_around_index(self, traces, index, share, leakage_values):

        for j in range(len(self.pattern)):
            traces[:, index + self.pattern[j]] += leakage_values[:, share] * 10

        return traces

    def create_pattern(self, num_points, spread):
        self.pattern = np.random.randint(spread * 2, size=num_points) - spread // 2
