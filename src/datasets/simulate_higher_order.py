import numpy as np
import random

def hw(input: np.uint32):
    out = 0
    temp = input
    for i in range(32):
        if temp % 2 == 1:
            out = out + 1
        temp = temp >> 1
    return out
vec_hw = np.vectorize(hw)

class SimulateHigherOrder():

    def __init__(self, args, order, num_traces, num_attack_traces,  num_informative_features, num_features) -> None:

        add_noise = args["sim_noise"]
        leakage_model = args["leakage_model"]
        
        self.order = order
        self.uni_noise=args["sim_vary_noise"]
        print(args["sim_vary_noise"], self.uni_noise)
        self.pick_leakage_spread(args["sim_leakage"])
        print(f"sim_noise={add_noise}")
        self.num_traces = num_traces
        self.n_profiling = num_traces
        self.n_attack = num_attack_traces
        self.num_features = num_features
        self.num_informative_features = num_informative_features
        self.num_leakage_regions = 2
        self.rsm_mask = args["dataset_target"] == "dpa_v42"
        self.noise = add_noise
        if num_features//(order+1)==num_informative_features:
            self.x_profiling, self.profiling_masks, self.profiling_shares  = self.only_informative(num_traces)
            self.x_attack, self.attack_masks, self.attack_shares  = self.only_informative(num_attack_traces)
        else:
            self.create_pattern(num_informative_features//self.num_leakage_regions, 20)
            indices = 20 * np.random.randint(num_features//20, size=self.num_leakage_regions * (self.order+1))
            self.x_profiling, self.profiling_masks, self.profiling_shares  = self.generate_traces(num_traces, indices)
            self.x_attack, self.attack_masks, self.attack_shares  = self.generate_traces(num_attack_traces, indices)

        self.profiling_labels = self.profiling_masks[:, order] if leakage_model == "ID" else vec_hw(self.profiling_masks[:, order])
        self.attack_labels = self.attack_masks[:, order] if leakage_model == "ID" else vec_hw(self.attack_masks[:, order])


    def generate_traces(self, num_traces, leakage_region_indices):

        masks = np.random.randint(16, size=(num_traces, self.order + 1), dtype =np.uint8)
        if self.rsm_mask:
            rsm_masks = np.random.choice([3, 12, 53, 58, 80, 95, 102, 105, 150, 153, 160, 175, 197, 202, 243, 252], size=num_traces)
            masks[:, 0] = rsm_masks
        shares = np.zeros((num_traces, self.order + 1), dtype=np.uint8)

        for i in range(self.order):
            shares[:, i] = masks[:, i]
        temp = masks[:, 0]
        for i in range(1, self.order+1):
            temp = temp ^ masks[:, i]
        shares[:, self.order] = temp

        leakage_values = self.leakages_spread(shares, self.num_features, num_traces)
        #leakage_values = vec_hw(shares)
        
        traces = np.random.normal(0, 2.5, size=(num_traces, self.num_features))
        # How to include the actual leakage values is maybe a problem.
        # Perhaps try to put leakages of specific value in clusters, because current implementation seems unrealistic and problematic.
        
        for i in range(self.order + 1):
            for j in range(self.num_leakage_regions):
                
                traces = self.include_leakage_around_index(traces, leakage_region_indices[i *self.num_leakage_regions + j], i, leakage_values)
        return traces, masks, shares 
    
    def leakages_spread(self, shares, num_points, num_traces):
        print("-------------------------------s")
        leakage_spread = np.zeros((self.order +1, num_points, num_traces))
        for share in range(self.order+1):
            value = shares[:, share].copy()
            for i in range(num_points):
  
                bits = [i for i in range(4)]
                if i > 24:
                    bits = [i for i in range(4, 8)] 
                #bits = [(i//4)%8]
                #print(bits)
                #bits=[4, 5, 6, 7]
                leakage = np.zeros_like(value)
                for j in bits:
                    leakage = leakage + ((value >> j) & 1)
                if len(bits) == 1:
                    leakage = leakage* 3
                leakage_spread[share, i, :] = leakage
        return leakage_spread
                

    def leakages_spread_real(self, shares, num_points, num_traces):
        print("------------real---s")
        leakage_spread = np.zeros((self.order +1, num_points, num_traces))
        sample_source = np.arange(0, 9)
        for share in range(self.order+1):
            value = shares[:, share].copy()
            for i in range(num_points):
                num_bits = np.random.randint(1, 5)
                bits = np.random.choice(sample_source, num_bits, replace=False)
                #bits = [(i//4)%8]
                #print(bits)
                #bits=[4, 5, 6, 7]
                leakage = np.zeros_like(value)
                for j in bits:
                    leakage = leakage + ((value >> j) & 1)
                # if len(bits) == 1:
                #     leakage = leakage* 3
                leakage_spread[share, i, :] = leakage
        return leakage_spread
    
    def leakage_spread_hw(self, shares, num_points, num_traces):
        print("-------------HW-------------s")
        leakage_spread = np.zeros((self.order +1, num_points, num_traces))
        for share in range(self.order+1):
            value = shares[:, share].copy()
            for i in range(num_points):
            
                bits = [j for j in range(8)]
                leakage = np.zeros_like(value)
                for j in bits:
                    leakage = leakage + ((value >> j) & 1)
                # if len(bits) == 1:
                #     leakage = leakage* 3
                leakage_spread[share, i, :] = leakage
        return leakage_spread
    
    def leakage_spread_bit(self, shares, num_points, num_traces):
        print("-------------------------------s")
        leakage_spread = np.zeros((self.order +1, num_points, num_traces))
        for share in range(self.order+1):
            value = shares[:, share].copy()
            for i in range(num_points):
                bits = [i//(num_points//8) % 8]
                print(bits)
                leakage = np.zeros_like(value)
                for j in bits:
                    print(j)
                    leakage = leakage + ((value >> j) & 1)
                # if len(bits) == 1:
                #     leakage = leakage* 3
                leakage_spread[share, i, :] = leakage
        return leakage_spread



    
    def only_informative(self, num_traces):
        print("Only informativce")
        
        masks = np.random.randint(256, size=(num_traces, self.order + 1), dtype =np.uint8)
        if self.rsm_mask:
            rsm_masks = np.random.choice([3, 12, 53, 58, 80, 95, 102, 105, 150, 153, 160, 175, 197, 202, 243, 252], size=num_traces)
            masks[:, 0] = rsm_masks
        shares = np.zeros((num_traces, self.order + 1), dtype=np.uint8)

        for i in range(self.order):
            shares[:, i] = masks[:, i]
        temp = masks[:, 0].copy()
        for i in range(1, self.order+1):
            temp = temp ^ masks[:, i]
        shares[:, self.order] = temp


       # traces = np.random.normal(0, 3, size=(num_traces, self.num_features))

        leakage_values =self.leakage_func(shares=shares, num_points=self.num_informative_features, num_traces=num_traces)
        if not self.uni_noise:
            traces = np.random.normal(0, self.noise, size=(num_traces, self.num_features))

            for i in range(self.order + 1):
                for j in range(self.num_informative_features):
                    traces[: , i*self.num_informative_features + j] += leakage_values[i, j, :]
        else:
            traces = np.zeros((num_traces, self.num_features))

            for i in range(self.order + 1):
                for j in range(self.num_informative_features):
                    traces[: , i*self.num_informative_features + j] += leakage_values[i, j, :]
                    temp = np.random.uniform(low=0, high=self.noise)
                    traces[: , i*self.num_informative_features + j] += np.random.normal(0, temp, size=(num_traces))
        
        return traces, masks, shares 
    
    def pick_leakage_spread(self, lm):
        if lm == "real":
            self.leakage_func = self.leakages_spread_real
        elif lm == "hw":
            self.leakage_func = self.leakage_spread_hw
        elif lm == "bit":
            self.leakage_func = self.leakage_spread_bit
        else:
            self.leakage_func = self.leakages_spread

    
    def include_leakage_around_index(self, traces, index, share, leakage_values):
        print(index)
        
        for j in range(len(self.pattern)):
            traces[:, index + self.pattern[j]] += leakage_values[share, j, :] * 10
        
        return traces

    def create_pattern(self, num_points, spread):
        self.pattern = np.random.randint(spread*2, size=num_points) - spread//2
    