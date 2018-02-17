import random
import numpy as np

#Activation function defintions
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z));
sigmoid_vec = np.vectorize(sigmoid);

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z));
sigmoid_prime_vec = np.vectorize(sigmoid_prime);

class multilayered_perceptron():

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [];
        self.weights = [];
        for index in range( len(sizes) - 1 ):
            num_row = sizes[index + 1];
            self.biases.append(np.random.randn(num_row, 1));
        for index in range( len(sizes) - 1 ):
            num_row = sizes[index + 1];
            num_col = sizes[index];
            self.weights.append( np.random.randn(num_row, num_col) );

    def feedforward(self, activation):
        for index in range( len(self.biases) ):
            bias = self.biases[index];
            weight = self.weights[index];
            activation = sigmoid_vec(np.dot(weight, activation) + bias);
        return activation

    def train(self, training_data, numIters, mini_batch_size, eta, testing_data=None):
        n = len(training_data)
        for iter_pos in range(numIters):
            print 'Starting iteration {}'.format(iter_pos);
            random.shuffle(training_data);

            #make mini_batches
            mini_batches =[];
            for data_index in range(0, n, mini_batch_size):
                this_batch = training_data[data_index:data_index + mini_batch_size];
                mini_batches.append(this_batch);

            #process the minibatches
            for mini_batch in mini_batches:
                self.process_mini_batch(mini_batch, eta);

    def process_mini_batch(self, mini_batch, eta):
        bias_update_term = [np.zeros(bias_vector.shape) for bias_vector in self.biases];
        weight_update_term = [np.zeros(weight_matrix.shape) for weight_matrix in self.weights];
        for feature, label in mini_batch:
            #calculate the mini_updates
            bias_mini_update, weight_mini_update = self.backpropagate(feature, label);
            for index in range( len(self.biases) ):
                bias_update_term[index] = bias_update_term[index] + bias_mini_update[index];
                weight_update_term[index] = weight_update_term[index] + weight_mini_update[index];
        #update the actual weights using the sum of the gradients of the examples in the batch
        for index in range( len(self.biases) ):
            self.weights[index] = self.weights[index] - (eta/len(mini_batch))*weight_update_term[index];
            self.biases[index] = self.biases[index] - (eta/len(mini_batch))*bias_update_term[index];

    def backpropagate(self, features, label):

        #gradients of the cost function in the same form as bias, weights
        snork_bias = [np.zeros(bias_vector.shape) for bias_vector in self.biases];
        snork_weight = [np.zeros(weight_matrix.shape) for weight_matrix in self.weights];
        #feedforward algorithm
        activation = features
        activations = [features] #store all the activations to backpropagate later
        z_list = [] #store the z vectors (dot(w, a)+b ones) to backpropagate later
        for index in range( len(self.biases) ):
            z_vector = np.dot(self.weights[index], activation) + self.biases[index];
            z_list.append(z_vector);
            activation = sigmoid_vec(z_vector); #update activations to feedforward
            activations.append(activation); 

        #calculate the delta for the last layer by performing the Hadamard Product
        delta_last_layer = self.cost_derivative(activations[len(activations)-1], label) * sigmoid_prime_vec(z_list[len(z_list)-1]);
        
        #the gradient for the bias is simply the delta
        snork_bias[len(snork_bias) - 1] = delta_last_layer;

        #the gradient for the weight is the dot product of the delta and the transpose of the earlier layer
        snork_weight[len(snork_weight) -1] = np.dot(delta_last_layer, activations[len(activations)-2].transpose());

        #set change_iter to the delta_last_layer, this is the variable that will change through iteration
        change_iter = delta_last_layer;

        #backpropagate all the way to the beginning, reverse index indicating index from the end
        for reverse_index in range(2, self.num_layers):
            current_z_vector = z_list[-reverse_index]
            current_act = sigmoid_prime_vec(current_z_vector)
            change_iter = np.dot(self.weights[-reverse_index + 1].transpose(), change_iter) * current_act
            snork_bias[-reverse_index] = change_iter
            snork_weight[-reverse_index] = np.dot(change_iter, activations[-reverse_index - 1].transpose())
        return (snork_bias, snork_weight)

    def classify(self, example):
        scores = self.feedforward(example[0]);
        classification = np.argmax(scores);
        return classification;

    def test(self, test_data):
        total_correct = 0;
        for i in range(len(test_data)):
            example = test_data[i];
            classification = self.classify(example);
            label = example[1];
            if label == classification:
                total_correct = total_correct + 1
        print 'Percent correct: {}'.format( float(total_correct)/len(test_data) )

    def cost_derivative(self, output_activations, y):
        #derivative of the quadratic cost function
        return (output_activations-y) 

