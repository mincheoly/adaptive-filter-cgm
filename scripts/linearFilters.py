################################################################################
# Authors: Min Cheol Kim, Christian Choe

# Description: Classes for linear filters

# Linear filters implemented so far:
#   - mu_LMS
#   - alpha_LMS
################################################################################

import numpy as np

# This class implements the good old LMS filter
class mu_LMS_filter:
    
    def __init__(self, training_set, mu_fraction=0.5, num_weight=5, bias=False, causal=True, delay=0):
        self.training_set = training_set
        self.num_weight = num_weight
        self.mu_fraction = mu_fraction
        self.trace_R = self.est_trace_R()
        self.mu = self.mu_fraction / self.trace_R
        self.bias = bias
        self.causal = causal
        self.delay = delay
        if bias:
            self.weight = np.zeros((self.num_weight+1))
        else:
            self.weight = np.zeros((self.num_weight))
    
    def reset_weight(self):
        self.weight = np.zeros(len(self.weight))

    def train_single_patient(self, patient_id, stop_half = False, train_cycles=5):
        W = self.weight # initalize weights
        weight_path = []
        error_path = []
        value = self.training_set[patient_id]
        cgm = value[:, 1]
        bgm = value[:, 2]
        
        # set start and end index for training, taking into account delay and causality
        start_idx = np.max([self.num_weight-1, self.delay])
        if self.causal:
            end_idx = len(cgm)-1
            start_idx = np.max([self.num_weight-1, self.delay])
        else:
            end_idx = len(cgm)-1-(self.num_weight/2)
            start_idx = np.max([self.num_weight/2, self.delay])
        predicted = np.zeros(len(cgm))
        predicted.fill(np.nan)
        #end_idx = len(cgm)-1 if self.causal else len(cgm)-1-(self.num_weight/2)
        for cycle in range(train_cycles):
            for i in range(start_idx, (end_idx - start_idx)/2):
                if self.causal:
                    indices = range(i-(self.num_weight-1),i+1)
                else:
                    half = self.num_weight/2 #type is int, so no decimals
                    indices = range(i-(self.num_weight-1-half),i+1+half)
                if self.bias:
                    x_tap = np.concatenate([np.array([1]),cgm[indices]])
                else:
                    x_tap = cgm[indices]
                error = bgm[i-self.delay] - np.dot(x_tap, W)
                W = W + 2*self.mu*error*x_tap
                weight_path.append(W)
                error_path.append(error)
        for i in range(   ((end_idx - start_idx)/2), end_idx+1):
            if self.causal:
                indices = range(i-(self.num_weight-1),i+1)
            else:
                half = self.num_weight/2 #type is int, so no decimals
                indices = range(i-(self.num_weight-1-half),i+1+half)
            if self.bias:
                x_tap = np.concatenate([np.array([1]),cgm[indices]])
            else:
                x_tap = cgm[indices]
            if i > start_idx:
                predicted[i] = np.dot(x_tap, W)
            error = bgm[i-self.delay] - np.dot(x_tap, W)
            weight_path.append(W)
            error_path.append(error)

        return (weight_path, predicted, error_path)
    
    def train(self, num_repeat = 10):
        for k in range(num_repeat):
            for key, value in self.training_set.iteritems():
                W = self.weight # initalize weights
                #value = np.array(value)
                cgm = value[:, 1]
                bgm = value[:, 2]
                
                # set start and end index for training, taking into account delay and causality
                start_idx = np.max([self.num_weight-1, self.delay])
                if self.causal:
                    end_idx = len(cgm)-1
                    start_idx = np.max([self.num_weight-1, self.delay])
                else:
                    end_idx = len(cgm)-1-(self.num_weight/2)
                    start_idx = np.max([self.num_weight/2, self.delay])
                    
                #end_idx = len(cgm)-1 if self.causal else len(cgm)-1-(self.num_weight/2)
                for i in range(start_idx, end_idx+1):
                    if self.causal:
                        indices = range(i-(self.num_weight-1),i+1)
                    else:
                        half = self.num_weight/2 #type is int, so no decimals
                        indices = range(i-(self.num_weight-1-half),i+1+half)
                    if self.bias:
                        x_tap = np.concatenate([np.array([1]),cgm[indices]])
                    else:
                        x_tap = cgm[indices]
                    error = bgm[i-self.delay] - np.dot(x_tap,W)
                    W = W + 2*self.mu*error*x_tap
                self.weight = W # update weights after each patient
    
    # test signal should be a 1d array
    def apply_filter(self, test_signal):
        filtered_signal = np.zeros(len(test_signal))
        filtered_signal.fill(np.nan)
        
        # set end idex for filtering, taking into causality
        if self.causal:
            end_idx = len(test_signal)-1
        else:
            end_idx = len(test_signal)-1-(self.num_weight/2)
        #end_idx = len(cgm)-1 if self.causal else len(cgm)-1-(self.num_weight/2)
        
        for i in range(self.num_weight-1, end_idx+1):
            if self.causal:
                indices = range(i-(self.num_weight-1),i+1)
            else:
                half = self.num_weight/2 #type is int, so no decimals
                indices = range(i-(self.num_weight-1-half),i+1+half)
            if self.bias:
                x_tap = np.concatenate([np.array([1]),test_signal[indices]])
            else:
                x_tap = test_signal[indices]
            filtered_signal[i] = np.dot(x_tap, self.weight)

        # apply delay
        filler = np.zeros(self.delay)
        filler.fill(np.nan)
        filtered_signal = np.concatenate([filtered_signal[self.delay:], filler])

        return filtered_signal
    
    def est_trace_R(self):
        s_sq_avg = np.zeros((self.num_weight))
        for key, value in self.training_set.iteritems():
            y = value[:, 1]
            s_sq = np.zeros((self.num_weight))
            for i in range(len(y)-self.num_weight+1):
                x = y[i:i+self.num_weight]
                s_sq = s_sq + x*x
            s_sq = s_sq/i
            s_sq_avg = s_sq_avg + s_sq
        s_sq_avg = s_sq_avg/len(self.training_set)
        return sum(s_sq_avg)

# This class implements alpha LMS
# Update equation found in Handout 9 in class notes
class alpha_LMS_filter:
    
    def __init__(self, training_set, alpha=0.01, num_weight=5, bias=False, causal=True, delay=0):
        self.training_set = training_set
        self.num_weight = num_weight
        self.alpha = alpha
        self.bias = bias
        self.causal = causal
        self.delay = delay
        if bias:
            self.weight = np.zeros((self.num_weight+1))
        else:
            self.weight = np.zeros((self.num_weight))
    
    def reset_weight(self):
        self.weight = np.zeros(len(self.weight))
       
    def train_single_patient(self, patient_id, stop_half = False, train_cycles=5):
        W = self.weight # initalize weights
        weight_path = []
        error_path = []
        value = self.training_set[patient_id]
        cgm = value[:, 1]
        bgm = value[:, 2]
        
        # set start and end index for training, taking into account delay and causality
        start_idx = np.max([self.num_weight-1, self.delay])
        if self.causal:
            end_idx = len(cgm)-1
            start_idx = np.max([self.num_weight-1, self.delay])
        else:
            end_idx = len(cgm)-1-(self.num_weight/2)
            start_idx = np.max([self.num_weight/2, self.delay])
        predicted = np.zeros(len(cgm))
        predicted.fill(np.nan)
        #end_idx = len(cgm)-1 if self.causal else len(cgm)-1-(self.num_weight/2)
        for cycle in range(train_cycles):
            for i in range(start_idx, (end_idx - start_idx)/2):
                if self.causal:
                    indices = range(i-(self.num_weight-1),i+1)
                else:
                    half = self.num_weight/2 #type is int, so no decimals
                    indices = range(i-(self.num_weight-1-half),i+1+half)
                if self.bias:
                    x_tap = np.concatenate([np.array([1]),cgm[indices]])
                else:
                    x_tap = cgm[indices]
                error = bgm[i-self.delay] - np.dot(x_tap, W)
                W = W + self.alpha*error/(np.linalg.norm(x_tap)**2)*x_tap
                weight_path.append(W)
                error_path.append(error)
        for i in range(   ((end_idx - start_idx)/2), end_idx+1):
            if self.causal:
                indices = range(i-(self.num_weight-1),i+1)
            else:
                half = self.num_weight/2 #type is int, so no decimals
                indices = range(i-(self.num_weight-1-half),i+1+half)
            if self.bias:
                x_tap = np.concatenate([np.array([1]),cgm[indices]])
            else:
                x_tap = cgm[indices]
            if i > start_idx:
                predicted[i] = np.dot(x_tap, W)
            error = bgm[i-self.delay] - np.dot(x_tap, W)
            weight_path.append(W)
            error_path.append(error)

        return (weight_path, predicted, error_path)
    def train(self, num_repeat = 10):
        for k in range(num_repeat):
            for key, value in self.training_set.iteritems():
                W = self.weight # initalize weights
                #value = np.array(value)
                cgm = value[:, 1]
                bgm = value[:, 2]
                
                # set start and end index for training, taking into account delay and causality
                start_idx = np.max([self.num_weight-1, self.delay])
                if self.causal:
                    end_idx = len(cgm)-1
                    start_idx = np.max([self.num_weight-1, self.delay])
                else:
                    end_idx = len(cgm)-1-(self.num_weight/2)
                    start_idx = np.max([self.num_weight/2, self.delay])
                    
                #end_idx = len(cgm)-1 if self.causal else len(cgm)-1-(self.num_weight/2)
                for i in range(start_idx, end_idx+1):
                    if self.causal:
                        indices = range(i-(self.num_weight-1),i+1)
                    else:
                        half = self.num_weight/2 #type is int, so no decimals
                        indices = range(i-(self.num_weight-1-half),i+1+half)
                    if self.bias:
                        x_tap = np.concatenate([np.array([1]),cgm[indices]])
                    else:
                        x_tap = cgm[indices]
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
            end_idx = len(test_signal)-1-(self.num_weight/2)
        #end_idx = len(cgm)-1 if self.causal else len(cgm)-1-(self.num_weight/2)
        
        for i in range(self.num_weight-1, end_idx+1):
            if self.causal:
                indices = range(i-(self.num_weight-1),i+1)
            else:
                half = self.num_weight/2 #type is int, so no decimals
                indices = range(i-(self.num_weight-1-half),i+1+half)
            if self.bias:
                x_tap = np.concatenate([np.array([1]),test_signal[indices]])
            else:
                x_tap = test_signal[indices]
            filtered_signal[i] = np.dot(x_tap, self.weight)

        # apply delay
        filler = np.zeros(self.delay)
        filler.fill(np.nan)
        filtered_signal = np.concatenate([filtered_signal[self.delay:], filler])
        
        return filtered_signal