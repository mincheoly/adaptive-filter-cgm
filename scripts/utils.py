################################################################################
# Authors: Min Cheol Kim, Christian Choe

# Description: Various utility functions

# Functions implemented so far:
#   - evaluate_filter
################################################################################

import numpy as np

# This function takes in a filter and a test set.
# It returns how much better or worse the filtered signal is 
# compared to the original
# a negative return value indicates a filter better than original signal
def evaluate_filter(filter_obj, test_set, metric='MARD'):
    gold_signal = np.zeros(0) # gold_standard signal across all patients
    filtered_signal = np.zeros(0) # filtered signal across all patients
    original_signal = np.zeros(0) # original cgm signal across all patients
    for idx in test_set.keys():
        gold_signal = np.concatenate([gold_signal, test_set[idx][:,2]])
        filtered_signal = np.concatenate([filtered_signal, filter_obj.apply_filter(test_set[idx][:,1])])
        original_signal = np.concatenate([original_signal, test_set[idx][:,1]])
    if metric is 'MARD':
        filtered_MARD = np.nanmean(np.absolute(gold_signal-filtered_signal)) / np.mean(gold_signal)
        original_MARD = np.nanmean(np.absolute(gold_signal-original_signal)) / np.mean(gold_signal)
        return (filtered_MARD - original_MARD, filtered_MARD, original_MARD)
    if metric is 'MSE':
        filtered_MSE = np.nanmean((gold_signal-filtered_signal)**2)
        original_MSE = np.nanmean((gold_signal-original_signal)**2)
        return (filtered_MSE - original_MSE, filtered_MSE, original_MSE)
    else:
        return -1