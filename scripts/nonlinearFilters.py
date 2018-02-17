################################################################################
# Authors: Min Cheol Kim, Christian Choe

# Description: Classes for nonlinear filters

# Linear filters implemented so far:
#   - alpha_Volterra_LMS
################################################################################

import numpy as np
import itertools
import scipy.misc

def multiply(elem, vec, order, term_list):
    if order == 0:
        term_list.append(elem)
        return
    if order == 1:
        for value in vec:
            term_list.append(elem*value)
        return
    for idx, value in enumerate(vec):
        multiply(elem*value, vec[idx:], order - 1, term_list)
    
def cross_term(vec, order):
    term_list = []
    vec = np.concatenate([np.array([1]), vec])
    for idx, elem in enumerate(vec):
        multiply(elem, vec[idx:], order-1, term_list)
    return np.array(term_list)

class alpha_LMS_Volterra_filter:
    
    def __init__(self, training_set, alpha=0.01, num_tap=5, order=2, bias=False, causal=True, delay=0):
        self.training_set = training_set
        self.alpha = alpha
        self.num_tap = num_tap
        self.order = order
        self.bias = bias
        self.causal = causal
        self.delay = delay

        # calculate number_weights needed for all the cross terms
        self.num_weight = len(cross_term(range(self.num_tap),self.order)) - 1
        if bias:
            self.weight = np.zeros((self.num_weight+1))
        else:
            self.weight = np.zeros((self.num_weight))
    
    def reset_weight(self):
        self.weight = np.zeros(len(self.weight))

    def get_num_weight(self):
        return int(self.num_weight)

    def train_single_patient(self, patient_id):
        W = self.weight # initalize weights
        weight_path = []
        value = self.training_set[patient_id]
        cgm = value[:, 1]
        bgm = value[:, 2]
        
        # set start and end index for training, taking into account delay and causality
        start_idx = np.max([self.num_tap-1, self.delay])
        if self.causal:
            end_idx = len(cgm)-1
            start_idx = np.max([self.num_tap-1, self.delay])
        else:
            end_idx = len(cgm)-1-(self.num_tap/2)
            start_idx = np.max([self.num_tap/2, self.delay])
        predicted = np.zeros(len(cgm))
        predicted.fill(np.nan)
        #end_idx = len(cgm)-1 if self.causal else len(cgm)-1-(self.num_weight/2)
        for i in range(start_idx, end_idx+1):
            if self.causal:
                indices = range(i-(self.num_tap-1),i+1)
            else:
                half = self.num_weight/2 #type is int, so no decimals
                indices = range(i-(self.num_tap-1-half),i+1+half)

            #expand x_tap to get all the cross terms
            x_cross = cross_term(cgm[indices], self.order)

            if self.bias:
                x_tap = x_cross
            else:
                x_tap = x_cross[1:]

            if i > start_idx:
                predicted[i] = np.dot(x_tap, W)
            error = bgm[i-self.delay] - np.dot(x_tap, W)
            W = W + self.alpha*error/(np.linalg.norm(x_tap)**2)*x_tap
            weight_path.append(W)
        return (weight_path, predicted)

    def train(self, num_repeat = 10):
        for k in range(num_repeat):
            for key, value in self.training_set.iteritems():
                W = self.weight # initalize weights
                #value = np.array(value)
                cgm = value[:, 1]
                bgm = value[:, 2]
                
                # set start and end index for training, taking into account delay and causality
                start_idx = np.max([self.num_tap-1, self.delay])
                if self.causal:
                    end_idx = len(cgm)-1
                    start_idx = np.max([self.num_tap-1, self.delay])
                else:
                    end_idx = len(cgm)-1-(self.num_tap/2)
                    start_idx = np.max([self.num_tap/2, self.delay])
                    
                #end_idx = len(cgm)-1 if self.causal else len(cgm)-1-(self.num_weight/2)
                for i in range(start_idx, end_idx+1):
                    if self.causal:
                        indices = range(i-(self.num_tap-1),i+1)
                    else:
                        half = self.num_weight/2 #type is int, so no decimals
                        indices = range(i-(self.num_tap-1-half),i+1+half)

                    #expand x_tap to get all the cross terms
                    x_cross = cross_term(cgm[indices], self.order)

                    if self.bias:
                        x_tap = x_cross
                    else:
                        x_tap = x_cross[1:]
                    error = bgm[i-self.delay] - np.dot(x_tap,W)
                    W = W + self.alpha*error/(np.linalg.norm(x_tap)**2)*x_tap
                self.weight = W # update weights after each patient
    
    # test signal should be a 1d array
    def apply_filter(self, test_signal):
        filtered_signal = np.zeros(len(test_signal))
        filtered_signal.fill(np.nan)
        
        # set end idex for filtering, taking into causality
        if self.causal:
            end_idx = len(test_signal)-1
        else:
            end_idx = len(test_signal)-1-(self.num_tap/2)
        #end_idx = len(cgm)-1 if self.causal else len(cgm)-1-(self.num_weight/2)
        
        for i in range(self.num_tap-1, end_idx+1):
            if self.causal:
                indices = range(i-(self.num_tap-1),i+1)
            else:
                half = self.num_weight/2 #type is int, so no decimals
                indices = range(i-(self.num_tap-1-half),i+1+half)

            #expand x_tap to get all the cross terms
            x_cross = cross_term(test_signal[indices], self.order)

            if self.bias:
                x_tap = x_cross
            else:
                x_tap = x_cross[1:]
            
            filtered_signal[i] = np.dot(x_tap, self.weight)

        # apply delay
        filler = np.zeros(self.delay)
        filler.fill(np.nan)
        filtered_signal = np.concatenate([filtered_signal[self.delay:], filler])
        
        return filtered_signal