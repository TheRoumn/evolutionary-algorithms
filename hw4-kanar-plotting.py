# use and edit this file to make all the plots you need - it is generally easier
# than plotting directly after the run of the algorithm

import utils

import matplotlib.pyplot as plt 

plt.figure(figsize=(12,8))
utils.plot_experiments('tsp', ['default',
                              #  'std-cx0.8-m0.2-ml10-pmx',
                               'std-cx0.5-m0.2-ml10-pmx',
                              #  'std-cx0.25-m0.2-ml10-pmx',
                              #  'std-cx0.8-m0.2-ml10-er',
                               'std-cx0.5-m0.2-ml10-er-swap',
                              #  'std-cx0.25-m0.2-ml10-er-swap',
                              #  'std-cx0.25-m0.2-ml10-er-invert',
                               'std-cx0.5-m0.2-ml10-er-invert',
                               'std-cx0.5-m0.4-ml10-er-invert',
                               'std-cx0.5-m0.5-ml10-pmx-opt',
                               'std-cx0.5-m0.5-ml10-er-opt',
                              #  'std-cx0.5-m0.5-ml10-er-opt',
                                     ])
plt.show()
 