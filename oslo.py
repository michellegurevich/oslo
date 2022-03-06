#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Complexity Project


# In[2]:


## 1 Implementation of the Oslo model


# In[3]:


# import libraries used below
import numpy as np
from numpy.random import choice
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import sem
from collections import OrderedDict, Counter
from tqdm import tqdm
import pickle


# In[4]:


#Logbin function from the python script provided on BB

################################################################################
# Max Falkenberg McGillivray
# mff113@ic.ac.uk
# 2019 Complexity & Networks course
#
# logbin230119.py v2.0
# 23/01/2019
# Email me if you find any bugs!
#
# For details on data binning see Appendix E from
# K. Christensen and N.R. Moloney, Complexity and Criticality,
# Imperial College Press (2005).
################################################################################

def logbin(data, scale = 1., zeros = False):
    """
    logbin(data, scale = 1., zeros = False)

    Log-bin frequency of unique integer values in data. Returns probabilities
    for each bin.

    Array, data, is a 1-d array containing full set of event sizes for a
    given process in no particular order. For instance, in the Oslo Model
    the array may contain the avalanche size recorded at each time step. For
    a complex network, the array may contain the degree of each node in the
    network. The logbin function finds the frequency of each unique value in
    the data array. The function then bins these frequencies in logarithmically
    increasing bin sizes controlled by the scale parameter.

    Minimum binsize is always 1. Bin edges are lowered to nearest integer. Bins
    are always unique, i.e. two different float bin edges corresponding to the
    same integer interval will not be included twice. Note, rounding to integer
    values results in noise at small event sizes.

    Parameters
    ----------

    data: array_like, 1 dimensional, non-negative integers
          Input array. (e.g. Raw avalanche size data in Oslo model.)

    scale: float, greater or equal to 1.
          Scale parameter controlling the growth of bin sizes.
          If scale = 1., function will return frequency of each unique integer
          value in data with no binning.

    zeros: boolean
          Set zeros = True if you want binning function to consider events of
          size 0.
          Note that output cannot be plotted on log-log scale if data contains
          zeros. If zeros = False, events of size 0 will be removed from data.

    Returns
    -------

    x: array_like, 1 dimensional
          Array of coordinates for bin centres calculated using geometric mean
          of bin edges. Bins with a count of 0 will not be returned.
    y: array_like, 1 dimensional
          Array of normalised frequency counts within each bin. Bins with a
          count of 0 will not be returned.
    """
    if scale < 1:
        raise ValueError('Function requires scale >= 1.')
    count = np.bincount(data)
    tot = np.sum(count)
    smax = np.max(data)
    if scale > 1:
        jmax = np.ceil(np.log(smax)/np.log(scale))
        if zeros:
            binedges = scale ** np.arange(jmax + 1)
            binedges[0] = 0
        else:
            binedges = scale ** np.arange(1,jmax + 1)
            # count = count[1:]
        binedges = np.unique(binedges.astype('uint64'))
        x = (binedges[:-1] * (binedges[1:]-1)) ** 0.5
        y = np.zeros_like(x)
        count = count.astype('float')
        for i in range(len(y)):
            y[i] = np.sum(count[binedges[i]:binedges[i+1]]/(binedges[i+1] - binedges[i]))
            # print(binedges[i],binedges[i+1])
        # print(smax,jmax,binedges,x)
        # print(x,y)
    else:
        x = np.nonzero(count)[0]
        y = count[count != 0].astype('float')
        if zeros != True and x[0] == 0:
            x = x[1:]
            y = y[1:]
    y /= tot
    x = x[y!=0]
    y = y[y!=0]
    return x,y


# In[5]:


class Site:
    def __init__(self, choice_parameters, h=0):
        self.h = h
        self.choice_parameters = choice_parameters
        self.threshold_slope = choice(**choice_parameters)

    def __add__(self, other):
        return self.h + other.h

    def __sub__(self, other):
        return self.h - other.h

    def add_grain(self):
        self.h = self.h + 1

    def reset_th(self):
        self.threshold_slope = choice(**self.choice_parameters)
        
    def topple_grain(self):
        self.h = self.h - 1
        self.reset_th()
        
    def reset(self):
        self.h = 0
        self.reset_th()


# In[6]:


class Pile:
    def __init__(self, length, probs, threshold_zs):
        choice_args = dict(a=threshold_zs, p=probs)
        self.length = length
        self.ava_size = 0
        self.lattice = np.array([Site(choice_args) for _ in range(length)])
        self.is_at_steady_state = False

    def reset(self):
        for site in self.lattice:
            site.reset()
        self.is_at_steady_state = False

    def get_heights(self):
        return [i.h for i in self.lattice]
    
    def get_pile_height(self):
        return self.lattice[0].h

    def get_threshold_slopes(self):
        return [i.threshold_slope for i in self.lattice]

    def find_unstable_site_indices(self):
        current_slopes = np.append(self.lattice[:-1] - self.lattice[1:], self.lattice[-1].h)
        return [i for i, site in enumerate(self.lattice) if current_slopes[i] > site.threshold_slope]
    
    def relax(self, site_index):
        self.lattice[site_index].topple_grain()
        self.ava_size = self.ava_size + 1
        stop_len = site_index + 1

        if stop_len == self.length:
            self.is_at_steady_state = True
            return
        self.lattice[stop_len].add_grain()

    def drop_grain(self, site_index=0):
        self.ava_size = 0
        self.lattice[site_index].add_grain()

        while True:
            unstable_site_indices = self.find_unstable_site_indices()
            if not unstable_site_indices:
                break
            for i in unstable_site_indices:
                self.relax(i)


# In[7]:


# define the values project will use for Oslo model algorithm
TH_SLOPES = (1, 2)
PROBS = (0.5, 0.5)

# define plot aesthetics
FS = 12


# In[8]:


# Tests


# In[9]:


oslo25 = Pile(25, PROBS, TH_SLOPES)
oslo25.get_threshold_slopes()


# In[10]:


PROBS_REGIME_2 = (0.0, 1.0)
oslo_regime2 = Pile(25, PROBS_REGIME_2, TH_SLOPES)
1 in oslo_regime2.get_threshold_slopes()


# In[11]:


set(oslo25.get_threshold_slopes())


# In[12]:


oslo16 = Pile(16, PROBS, TH_SLOPES)
oslo32 = Pile(32, PROBS, TH_SLOPES)


# In[13]:


heights_L16 = []

for i in range(5000):
    oslo16.drop_grain()
    heights_L16.append(oslo16.get_pile_height())
 
np.average(heights_L16)


# In[14]:


heights_L32 = []

for i in range(15000):
    oslo32.drop_grain()
    heights_L32.append(oslo32.get_pile_height())
    
np.average(heights_L32)


# In[15]:


sem(heights_L16)


# In[16]:


sem(heights_L32)


# In[17]:


## 2 The height of the pile $h(t;L)$


# In[18]:


oslo4 = Pile(4, PROBS, TH_SLOPES)
oslo8 = Pile(8, PROBS, TH_SLOPES)
oslo16 = Pile(16, PROBS, TH_SLOPES)
oslo32 = Pile(32, PROBS, TH_SLOPES)
oslo64 = Pile(64, PROBS, TH_SLOPES)
oslo128 = Pile(128, PROBS, TH_SLOPES)
oslo256 = Pile(256, PROBS, TH_SLOPES)
oslo512 = Pile(512, PROBS, TH_SLOPES)


# In[19]:


piles_set = (oslo4, oslo8, oslo16, oslo32, oslo64, oslo128, oslo256, oslo512)
data_dict = OrderedDict()


# In[20]:


iterations = 200000


# In[21]:


# for pile in piles_set:
#     pile_info = {'h': [], 'asz': []}
#     while not pile.is_at_steady_state:
#         pile_info['h'].append(pile.get_pile_height())
#         pile_info['asz'].append(pile.ava_size)
#         pile.drop_grain()
#     for i in tqdm(range(iterations)):
#         pile_info['h'].append(pile.get_pile_height())
#         pile_info['asz'].append(pile.ava_size)
#         pile.drop_grain()
#     data_dict[pile.length] = pile_info
# pickle.dump(data_dict, open('data_file', 'wb'))


# In[22]:


data_dict = pickle.load(open('data_file', 'rb'))
steady_state_time_period = iterations


# In[23]:


for length, pile_info in data_dict.items():
    plt.plot(pile_info['h'], ':', label=length)

plt.title("Pile height $ h(t; L) $")
plt.xlabel("$ t $", fontsize=FS)
plt.xlim(0, 275000)
plt.ylabel("$ h $", fontsize=FS)
plt.legend(loc=4, title="System size $ L $")
plt.show()


# In[24]:


def np_moving_average(data, temporal_window=50):
    window = np.ones(temporal_window) / temporal_window
    return np.convolve(data, window, 'valid')


# In[25]:


t_range = 50
t_0 = int(t_range / 2)

for length, pile_info in data_dict.items():
    smooth_data = np_moving_average(pile_info['h'], t_range)
    times = np.arange(t_0, len(smooth_data) + t_0)
    
    # plot results over a log-log scale
    plt.loglog(times, smooth_data, ':', label = length)

# plot set-up
plt.title("Pile height $h(t; L) $")
plt.xlabel("$t$", fontsize=FS)
plt.ylabel("$h$", fontsize=FS)
plt.xlim(10e0, 10e6)
plt.legend(loc=4, title="System Size $ L $")
plt.show()


# In[26]:


for length, pile_info in data_dict.items():
    # use moving average method from previouly
    smooth_h = np_moving_average(pile_info['h'], t_range)
    
    # collapse averages
    smooth_h_collapse = smooth_h / length
    t_collapse = np.arange(t_0, len(smooth_h_collapse) + t_0) / length**2
    
    # plot results of data collapse
    plt.loglog(t_collapse, smooth_h_collapse, label=length)
    
# proportionality relationship for axes   
plt.title("Data collapse of smoothed pile heights") 
plt.xlabel("$ t/L^2 $", fontsize=FS)
plt.ylabel("$ \widetilde h/L $", fontsize=FS)
plt.legend(title="System Size $ L $")
plt.show()


# In[27]:


# for a function variable t, a constant coefficient a, and an exponent (power) k
power_law = lambda t, a, k: a * t ** k


# In[28]:


t_0 = 1000
h_values = data_dict[64]['h'][t_0:-steady_state_time_period]

offset_len = t_0 + len(h_values)
t = np.arange(t_0, offset_len)

(a, k), covm = curve_fit(power_law, t, h_values)
a, k, np.sqrt(np.diag(covm))


# In[29]:


# and in the same graph plot the data for h_tilde
plt.plot(t, h_values, label='Smoothed data', color='orange')

# plot the power law fit
plt.plot(t, power_law(t, a, k), label='Power law fit', color='red')

plt.title("Power law fit of data for smoothed pile heights")
plt.xlabel("Times $t$", fontsize=FS)
plt.ylabel("Heights $h$", fontsize=FS)
plt.show()


# In[30]:


# plot the relative difference of the fit and data
diff = power_law(t, a, k) - h_values
rel_diff = diff / h_values
# plt.plot(t, abs(rel_diff), color='green')
plt.plot(t, rel_diff, color='green')

plt.title("Relative difference between height data and fit")
plt.xlabel("$t$", fontsize=FS)
plt.ylabel("Percent difference", fontsize=FS)
plt.ylim(-1, 1)

plt.show()


# In[31]:


lengths = np.array(list(data_dict.keys()))


# In[32]:


# calculate average heights
avg_h = [np.average(pile_info['h'][-steady_state_time_period:]) for pile_info in data_dict.values()]

plt.plot(lengths, avg_h, '.', color='blue')

plt.xlabel("$ L $", fontsize=FS)
plt.ylabel(r"$ \langle h \rangle $", fontsize=FS)
plt.show()


# In[33]:


# calculate standard deviations
std_devs = [np.std(pile_info['h'][-steady_state_time_period:]) for pile_info in data_dict.values()]

# plot results over a log-log scale
plt.loglog(lengths, std_devs, '.', color='blue', label='data')

plt.xlabel("$ L $", fontsize=FS)
plt.ylabel("$ \sigma_h(L) $", fontsize=FS)
plt.xlim(10, 10e2)
plt.ylim(0, 10e0)
plt.show()


# In[34]:


def calculate_height_probability(height_data):
    total_time = len(height_data)
    height_frequencies = sorted(Counter(height_data).items())
    height_probabilities = OrderedDict()
    for (key, value) in height_frequencies:
        height_probabilities[key] = value / total_time
    return height_probabilities


# In[35]:


for length, pile_info in data_dict.items():
    h_values = pile_info['h'][-steady_state_time_period:]
    h_prob = calculate_height_probability(h_values)
    heights = list(h_prob.keys())
    probs = list(h_prob.values())
    plt.plot(heights, probs, '-', label=length, linewidth=0.5)
    
plt.title("Height probability $P(h;L)$ for systems $L_i$")
plt.legend(title="System Size (L)",  framealpha=0.8, prop={'size':10})
plt.xlabel("$ h $", fontsize=FS)
plt.ylabel("$ P(h;L) $", fontsize=FS)
plt.xlim(-100,1000)
plt.ylim(0,.5)
plt.show()


# In[36]:


calc_avg_h = lambda L, a_0, a_1, om_1: a_0 * L * (1 - a_1 * L ** (-om_1))
(a_0, a_1, om_1), covm = curve_fit(calc_avg_h, lengths, avg_h, absolute_sigma=True)
a_0, a_1, om_1


# In[37]:


a = 0.58
om = 0.24
range_l = np.arange(1, 512)
plt.loglog(lengths, std_devs, '.', color='blue', label='Data for system size $L=512$')
plt.loglog(range_l, power_law(range_l, a, om), color='black', label='Power law fit with $a = 0.58, \omega=0.25$')

plt.title("Scaled $\sigma_h(L) $ and its power law approximation")
plt.xlabel("$ L $", fontsize=FS)
plt.ylabel("$ \sigma_h(L) $", fontsize=FS)
plt.xlim(10, 10e2)
plt.ylim(10e-2, 10e0)
plt.legend(loc=4)
plt.show()


# In[38]:


(a, w), covm = curve_fit(power_law, lengths, std_devs)

a, w, np.sqrt(np.diag(covm))

for length, pile_info in data_dict.items():
    height_prob_dict = calculate_height_probability(pile_info['h'][-steady_state_time_period:])
    collapsed_h = (np.array(list(height_prob_dict.keys()))-avg_h[list(data_dict.keys()).index(length)]) / length ** w 
    collapsed_p = np.array(list(height_prob_dict.values())) * length ** w
    plt.plot(collapsed_h, collapsed_p, label=length, linewidth=0.5)

plt.title("Data collapse of measured height probability $P(h;L)$")
plt.legend(title="System Size (L)")
plt.xlabel(r"$(h - \langle h \rangle) L^{-0.24}$", fontsize=FS)
plt.ylabel("$L^{0.24}P(h; L)$", fontsize=FS)
plt.xlim(-3,3)
plt.ylim(0,.8)

plt.show()


# In[39]:


def make_log_bins(data):
    centers, probabilities = logbin(data, scale=1.2, zeros=True)
    return np.array(centers), np.array(probabilities)


# In[40]:


ava_size_256 = data_dict[256]['asz']

no_of_samples = 1000000
ava_size_prob_dict = calculate_height_probability(ava_size_256[-i:])
plt.loglog(list(ava_size_prob_dict.keys()), list(ava_size_prob_dict.values()), '.', ms=1.5, label="$N=1000000$", color="blue")
centers, probs = make_log_bins(ava_size_256[-i:])
plt.loglog(centers, probs, '-', color="pink")
plt.xlabel("$s$", fontsize=FS)
plt.ylabel("$P_N(s;L)$", fontsize=FS)
plt.legend(loc=3)
plt.xlim(10e-1,10e5)
plt.ylim(10e-10,10e-1)
plt.show()


# In[41]:


no_of_samples = 1000000

for length, pile_info in data_dict.items():
    centers, probabilities = make_log_bins(pile_info['asz'][-no_of_samples:])
    plt.loglog(centers, probabilities, '-', label=length, linewidth=0.5)

plt.legend(loc=3, title='System Size (L)', prop={'size':10})
plt.xlabel("$s$", fontsize=14)
plt.ylabel(r"$\widetilde P_N(s;L)$", fontsize=14)
plt.show()


# In[42]:


centers, probabilities = make_log_bins(data_dict[512]['asz'][-no_of_samples:])
power_law = lambda s, a, tau_s: a * s ** tau_s

(a, tau_s), cov = curve_fit(power_law, centers[22:-10], probabilities[22:-10])

a, tau_s, np.sqrt(np.diag(cov))


# In[43]:


for length, pile_info in data_dict.items():
    centers, probabilities = make_log_bins(pile_info['asz'][-no_of_samples:])
    s_tau_probabilities = centers ** -tau_s * probabilities
    plt.loglog(centers, s_tau_probabilities, '-', label=length, linewidth=0.5)

plt.legend(loc=0, title='System Size (L)', prop={'size':10})
plt.xlabel("$s$", fontsize=14)
plt.ylabel(r"$s^{\tau_s} \widetilde P_N(s;L)$", fontsize=14)
plt.show()


# In[44]:


D = 2.25

for length, pile_info in data_dict.items():
    centers, probabilities = make_log_bins(pile_info['asz'][-no_of_samples:])
    center_L_minus_Ds = centers / (length ** D)
    s_tau_probabilities = centers ** -tau_s * probabilities
    plt.loglog(center_L_minus_Ds, s_tau_probabilities, '-', label=length, linewidth=0.5)
        
plt.legend(title='System size $L$')
plt.xlabel(r"$s/L^{(D = 2.25)}$", fontsize=FS)
plt.ylabel(r"$s^{(\tau_s = 1.55)} \cdot \widetilde P_N(s;L)$", fontsize=14)
plt.xlim(10e-4,10e0)
plt.show()


# In[45]:


### Task 3b: Measuring directly the $ k $th moment $ \langle s_k \rangle $


# In[46]:


k_set = (1, 2, 3, 4)
kth_moment_list = []

for k in k_set:
    kth_moments = []
    for length, pile_info in data_dict.items():
        kth_moment = np.average(np.array(pile_info['asz'][-steady_state_time_period:], dtype='float64') ** k)
        kth_moments.append([length, kth_moment])
    kth_moment_list.append(kth_moments)
    
kth_moment_array = np.array(kth_moment_list)


# In[47]:


for i, k in enumerate(k_set):
    plt.loglog(kth_moment_array[i, :, 0], kth_moment_array[i, :, 1], '.', label='k = {}'.format(k))

plt.xlabel('$ L $', fontsize=FS)
plt.ylabel(r'$\langle s^k \rangle$', fontsize=FS)
plt.show()


# In[48]:


linear_regression = lambda x, c, m: m * x + c

Ls = np.arange(0, 1000)
k_slopes = []
k_slope_errs = []

for i, k in enumerate(k_set):
    plt.loglog(kth_moment_array[i, :, 0], kth_moment_array[i, :, 1], '.', label='k = {}'.format(k))
    (log_a, exp), cov = curve_fit(linear_regression, np.log(kth_moment_array[i, -3:, 0]), np.log(kth_moment_array[i, -3:, 1]))
    plt.loglog(Ls, power_law(Ls, np.exp(log_a), exp), '--', color='black', linewidth=0.8)
    k_slopes.append(exp)
    k_slope_errs.append(np.sqrt(np.diag(cov)[1]))

plt.legend(loc=0)
plt.xlabel('$L$', fontsize=14)
plt.ylabel(r'$\langle s^k \rangle$', fontsize=FS)
plt.ylim(10e-5,10e22)
plt.xlim(10e-1,10e2)
plt.show()


# In[49]:


ks = (1, 2, 3, 4)
(c, D), cov = curve_fit(linear_regression, ks, k_slopes, sigma=k_slope_errs)
print(c, D, np.diag((cov)[1]))

tau_s = 1 - c / D
print(tau_s)

plt.errorbar(ks, k_slopes, yerr=k_slope_errs, color='red', fmt='.', label='data', ms=2.0)

ks_ = np.arange(6)
plt.plot(ks_, linear_regression(ks_, c, D), color='orange', label='fit', linewidth=0.5)

plt.legend(loc=0)
plt.xlabel('$k$', fontsize=14)
plt.ylabel(r'$\phi$', fontsize=14)
plt.show()


# In[ ]:




