# use and edit this file to make all the plots you need - it is generally easier
# than plotting directly after the run of the algorithm

import utils

import matplotlib.pyplot as plt 

plt.figure(figsize=(12,8))
utils.plot_experiments('partition', ['default',
                                    #  'cx0.8-m0.2-f0.4-e0.1',
                                    #  'cx0.9-m0.1-f0.3',
                                    #  'cx0.8-m0.5-f0.05-e0.1',
                                    #  'cx0.8-m0.5-f0.1-e0.2',
                                    #  'cx0.8-m0.5-f0.05-e0.05',
                                    #  'cx0.9-m0.7-f0.1-e0.05',
                                    #  'cx0.9-m0.7-f0.3-e0.05',
                                    #  'cx0.8-m0.5-f0.05-e0.05-g1000-std',
                                    #  'cx0.8-m0.5-f0.2-e0.15-g1000-std',
                                    #  'cx0.8-m0.75-f0.05-e0.15-g1000-std',
                                    #  'cx0.8-m0.75-f0.05-e0.05-g1000-std',
                                    #  'cx0.65-m0.5-f0.07-e0.07-g500-std',
                                    #  'cx0.7-m0.5-f0.07-e0.07-g500-std',
                                    #  'cx0.7-m0.5-f0.07-e0.03-g500-std-sus',
                                    #  'cx0.7-m0.5-f0.07-e0.03-g500-std-sus-lightest-first',
                                    #  'cx0.7-m0.5-f0.05-e0.05-g500-std-sus-no-dups',
                                    #  'cx0.5-m0.3-f0.05-e0.05-g500-std-rule-no-dups',
                                    # #  'cx0.4-m0.4-f0.05-e0.05-g500-std-rule-no-dups',
                                    #  'cx0.6-m0.2-f0.05-e0.02-g500-std-rule-no-dups',
                                    #  'cx0.8-m0.3-f0.05-e0.03-g500-std-rule-no-dups',
                                    #  'cx0.8-m0.3-f0.05-e0.03-g500-std-rule-no-dups-2',
                                    #  'cx0.8-m0.35-f0.1-e0.02-g1000-std-rule-no-dups',
                                    #  'cx0.8-m0.5-f0.075-e0.05-g1000-std-rule-no-dups',
                                    #  'cx0.8-m0.3-f0.05-e0.03-g1000-std-rule-no-dups',
                                    #  'cx0.8-m0.5-f0.08-e0.1-g1000-std-rule-no-dups',
                                    #  'cx0.8-m0.5-f0.15-e0.1-g1000-std-rule-no-dups',
                                    #  'cx0.8-m0.5-f0.08-e0.1-g5000-std-rule-no-dups',
                                    #  'cx0.8-m0.4-f0.004-e0.05-G1000-std-tour-no-dups',
                                    #  'cx0.6-m0.6-f0.004-e0.05-G1000-std-tour-no-dups',
                                    #  'cx0.4-m0.6-f0.004-e0.1-G1000-std-tour-no-dups',
                                    #  'cx0.25-m0.8-f0.004-e0.1-G1000-std-tour-no-dups',
                                    #  'cx0.5-m0.8-f0.004-e0.1-G1000-std-tour-no-dups',
                                    #  'cx0.7-m0.5-f0.004-e0.07-G1000-std-tour-no-dups',
                                    #  'cx0.7-m0.6-f0.004-e0.05-G1000-std-tour-no-dups',
                                    #  'cx0.4-m0.65-f0.004-e0.15-G1000-std-tour-no-dups',
                                    #  'cx0.4-m0.65-f0.004-e0.15-G1000-std-rule-no-dups',
                                    #  'cx0.4-m0.65-f0.004-e0.05-G1000-std-rule-no-dups',
                                    #  'cx0.4-m0.8-f0.004-e0.05-G2000-diff-rule-no-dups',
                                    #  'cx0.3-m0.6-f0.004-e0.05-G2000-diff-rule-no-dups',
                                    #  'cx0.3-m0.8-f0.004-e0.1-G2000-diff-rule-no-dups',
                                    #  'cx0.35-m0.9-f0.004-e0.07-G2000-diff-rule-no-dups-minus-min',
                                    #  'cx0.35-m0.7-f0.004-e0.04-G2000-diff-rule-no-dups-minus-min',
                                    #  'cx0.35-m0.55-f0.004-e0.04-G2000-diff-rule-no-dups-minus-min',
                                    #  'cx0.35-m0.75-f0.004-e0.07-G4000-diff-rule-no-dups-minus-min',
                                    #  'cx0.35-m0.9-f0.004-e0.1-G4000-diff-rule-no-dups-minus-min',
                                    #  'cx0.35-m0.8-f0.004-e0.05-G4000-diff-rule-no-dups-minus-min',
                                    #  'cx0.35-m0.75-f0.004-e0.07-G8000-diff-rule-no-dups-minus-min', # sub 50
                                    #  'cx0.35-m0.75-f0.004-e0.07-G8000-diff-rule-no-dups-minus-min'
                                    #  'cx0.35-m0.75-f0.004-e0.1-G16000-diff-rule-no-dups' # also sub 50
                                     'cx0.35-m0.75-f0.004-e0.05-G4000-diff-rule-smart-s1',
                                     'cx0.35-m0.75-f0.004-e0.05-G4000-diff-rule-smart2-s1'
                                     ])
plt.show()
 