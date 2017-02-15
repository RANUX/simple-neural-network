import matplotlib.animation as animation
import matplotlib.pyplot as plt
from numpy import array, dot, exp, random
import numpy as np
import math

fig, ax = plt.subplots()

class NeuralNetwork():
    def __init__(self):
        # Seed the random number generator, so it generates the same numbers
        # every time the program runs.
        random.seed(1)

        # We model a single neuron, with 3 input connections and 1 output connection.
        # We assign random weights to a 3 x 1 matrix, with values in the range -1 to 1
        # and mean 0.
        self.synaptic_weights = 2 * random.random((2, 1)) - 1

    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        self.training_set_inputs = training_set_inputs

        for iteration in xrange(number_of_training_iterations):
            # Pass the training set through our neural network (a single neuron).
            # output is converted to sigmoid
            output = self.think(training_set_inputs)

            # Calculate the error (The difference between the desired output
            # and the predicted output).
            error = training_set_outputs - output

            # print mean squared error
            print "error: %f" % math.sqrt((error**2).sum(axis=0)[0]/ len(training_set_outputs))

            # Multiply the error by the input and again by the gradient of the Sigmoid curve.
            # This means less confident weights are adjusted more.
            # This means inputs, which are zero, do not cause changes to the weights.

            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))
            self.plot(iteration)

            # Adjust the weights.
            self.synaptic_weights += adjustment

    # The neural network thinks.
    def think(self, inputs, debug=False):
        # Pass inputs through our neural network (our single neuron).
        dotProd = dot(inputs, self.synaptic_weights)
        if debug:
            print "inputs: "+inputs.__str__()
            print "synaptic_weights: "+self.synaptic_weights.__str__()
            print "dot: "+dotProd.__str__()
        return self.__sigmoid(dotProd)

    def plot(self, t):
        x = [self.__sigmoid(self.synaptic_weights[0][0]), self.training_set_inputs[0][0]]
        y = [self.__sigmoid(self.synaptic_weights[1][0]), self.training_set_inputs[0][1]]
        if t == 0:
            self.points, = ax.plot(x, y, marker='o', linestyle='None')
            ax.set_xlim(-2, 6)
            ax.set_ylim(-2, 6)
        else:
            self.points.set_data(x, y)
        plt.pause(0.01)


if __name__ == "__main__":

    #Intialise a single neuron neural network.
    neural_network = NeuralNetwork()

    print "Random starting synaptic weights: "
    print neural_network.synaptic_weights

    # The training set. We have 4 examples, each consisting of 3 input values
    # and 1 output value.
    #training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    #training_set_outputs = array([[0, 1, 1, 0]]).T

    # xor
    training_set_inputs  = array([[1, 0], [0, 1], [1, 1], [0, 0]])
    training_set_outputs = array([[1, 0, 0, 0]]).T

    # Train the neural network using a training set.
    # Do it 10,000 times and make small adjustments each time.
    neural_network.train(training_set_inputs, training_set_outputs, 500)

    print "New synaptic weights after training: "
    print neural_network.synaptic_weights

    # Test the neural network with a new situation.
    print "Considering new situation [1, 0] -> ?: "
    print neural_network.think(array([1, 0]))
