# use and edit this file to make all the plots you need - it is generally easier
# than plotting directly after the run of the algorithm

import utils

import matplotlib.pyplot as plt
selected_functions = ['f01', 'f02', 'f06', 'f08', 'f10']
legend_names = {'f01': 'Sphere', 'f02': 'Ellipsoidal', 'f06': 'Attractive Sector', 'f08': 'Rosenbrock', 'f10': 'Rotated Ellipsoidal'}
wanted_exp_ids = ['default',
                #   'linear_decay',
                  'exp_decay',
                #   'arithmetic_cx_exp_decay',
                #   'adaptive_arithmetic_cx_exp_decay',
                #   'adaptive_arithmetic_cx_minus_min_exp_decay',
                #       'one_pt_diff',
                  'differential_complete',
                  'differential_complete_exp_decay',
                  'differential_complete_random',
                  'differential_complete_random_k2',
                  'differential_complete_random_k3',
                  ]
for function in selected_functions:
    legend_name = legend_names[function]
    rename_dict = { f'{exp_id}.{function}' : f'{exp_id}.{legend_name}' for exp_id in wanted_exp_ids}
    to_plot = list(rename_dict.keys())
    plt.figure(figsize=(12,8))
    utils.plot_experiments('continuous', to_plot)
    plt.yscale('log')
    # plt.savefig(f'figures/hw3-{function}-part2_diff.png')
    plt.show()
 