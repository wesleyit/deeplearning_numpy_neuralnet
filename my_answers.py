"""
WS-DeepNet - A Multilayer Perceptron in Python
==============================================

Hi! This is an Multilayer Perceptron implementation in Python,
using Numpy as helper for some math operations.
"""

import numpy as np

class NeuralNetwork(object):
    """
    Usage
    -----

    Import this class and use its constructor to inform:

     - How many input nodes this net should have (int)
     - How many hidden layers you want (int)
     - How many output nodes you need (int)
     - What is the desired learning rate (float)

    Example:
    ```
    from my_answers import NeuralNetwork
    nn = NeuralNetwork(3, 2, 1, 0.98)
    ```
    This will create a neural network object into the nn
    variable with 3 input layers, 2 hidden layers and
    1 output, using 0.98 as learning rate.
    """

    def __init__(self, input_nodes, hidden_nodes,
                 output_nodes, learning_rate):
        """
        The default behavior is to use:

         - 32 hidden nodes
         - 1 output nodes
         - learning_rate = 0.98
         - 3000 iterations

        Set your values during the network creation, like:

        `nn = NeuralNetwork(3, 2, 1, 0.98)`

        """

        # Hyperparameters from default or from constructor
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize the weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5,
                                                        (self.input_nodes, self.hidden_nodes))
        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5,
                                                         (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate

        # The default activation function is a sigmoid(x)
        self.activation_function = lambda x: 1 / (1 + np.exp(-x))


    def train(self, features, targets):
        """
        Train the network on batch of features and targets.

        Arguments
        ---------

        features: 2D array, each row is one data record, each column is a feature
        targets: 1D array of target values
        """

        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            final_outputs, hidden_outputs = self.forward_pass_train(X)
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y,
                                                                        delta_weights_i_h, delta_weights_h_o)
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)


    def forward_pass_train(self, X):
        """
        This is the feedforward step used during
        the training process.

        Arguments
        ---------
        X: features batch
        """

        # This is where the inputs feed the hidden layer through the first
        # layer of weights.
        hidden_inputs = np.dot(X, self.weights_input_to_hidden)
        hidden_outputs = self.activation_function(hidden_inputs)

        # This is the final flow. We are going to use a linear output
        # here instead of the sigmoid function.
        # The linear output is represented by f(x) = x.
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)
        final_outputs = final_inputs

        # Return final_outputs and hidden_outputs, to make it easier to train
        # the network duting the backpropagation steps.
        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs,
                        X, y, delta_weights_i_h, delta_weights_h_o):
        """
        Implement backpropagation

        Arguments
        ---------
        final_outputs: output from forward pass
        y: target (i.e. label) batch
        delta_weights_i_h: change in weights from input to hidden layers
        delta_weights_h_o: change in weights from hidden to output layers
        """

        # The error of the output layer.
        # Its formula is E = y - ŷ, where y is the label and ŷ is the
        # output given after the feedforward step.
        error = y - final_outputs

        # Since we are using a linear function, the output error term is:
        output_error_term = error

        # The hidden layer's contribution to the error
        # and its gradient.
        hidden_error = np.dot(self.weights_hidden_to_output, output_error_term)
        hidden_error_term = hidden_error * hidden_outputs * (1 - hidden_outputs)

        # Weight step (input to hidden)
        delta_weights_i_h += hidden_error_term * X[:, None]

        # Weight step (hidden to output)
        delta_weights_h_o += np.dot(hidden_outputs.reshape(hidden_outputs.shape[0],
                                                           output_error_term.shape[0]),
                                    output_error_term.reshape([1, 1]))

        # Return the delta to update weights from i_h and h_o layers.
        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        """
        Update weights on gradient descent step

        Arguments
        ---------
        delta_weights_i_h: change in weights from input to hidden layers
        delta_weights_h_o: change in weights from hidden to output layers
        n_records: number of records
        """

        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records
        self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records


    def run(self, features):
        """
        Run a forward pass through the network with input features

        Arguments
        ---------
        features: 1D array of feature values
        """

        # The same steps as in the forward_pass_train.
        hidden_inputs = np.dot(features, self.weights_input_to_hidden)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)
        final_outputs = final_inputs
        return final_outputs

# Hyperparameters
hidden_nodes = 16
output_nodes = 1
learning_rate = 0.98
iterations = 3000
