#!/usr/bin/python
import numpy as np
from random import seed
from random import random
from math import exp

# Purpose: Create neural network with random weights between 0 and 1
# Arguments:
#	  num_inputs: number of input neurons
#	  num_hidden: number of hidden neurons
#	  num_outpus: number of output neurons
def initialize_network(num_inputs, num_hidden, num_outputs):
	# network is a list of layers
	# layers are a list of neurons
	# neuron is a dictionary with a list 'weights' of input synapse weights
	network = list()
	hidden_layer = [{
						'weights': [random() for i in range(num_inputs + 1)
					]} for i in range(num_hidden)]

	network.append(hidden_layer)
	output_layer = [{
						'weights':[random() for i in range(num_hidden + 1)
					]} for i in range(num_outputs)]

	network.append(output_layer)
	return network


# Purpose: Find input to neuron
# Arguments: 
#	  weights: synapse weights that lead to neuron
#	   inputs: data being sent through synapses
def activate(weights, inputs):
	activation = weights[-1] #bias
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation


# Purpose: Sigmoid transfer function between layers
# Arguments:
#  activation: output of neuron
def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))


# Purpose: Find approx hot encoding for expected iris type
# Arguments:
#	  network: network to update weights in
#	      row: iris data to determine type of
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs #move to next layer, outputs become inputs
	return inputs


# Purpose: derivative of sigmoid function
# Arguments:
#	  output: arg for deriv of sigmoid function
def transfer_derivative(output):
	return output * (1.0 - output)


# Purpose: Backward propagate expected values to generate deltas in network
# Arguments:
#	  network: network to update weights in
#	 expected: the expected output of the network
def backward_propagate_error(network, expected):
	# iterate through layers starting at output layer
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()

		if i != len(network) - 1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:
			# get list of differences between output neurons and expected output
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(expected[j] - neuron['output'])
		# place delta for each neuron within neuron
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


# Purpose: Update the weights of synapses in network
# Arguments:
#	  network: network to update weights in
#	      row: example from training data that was used to generate errors
#	   l_rate: the learning rate to use to modify weights
def update_weights(network, row, l_rate):
	# iterate through layers of network
	for i in range(len(network)):
		inputs = row[:-1] 	
		
		# set the inputs to be the outputs of the current layer after updating 
		# 	current layer weights
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]

		#iterate through neurons in each layer
		for neuron in network[i]:
			for j in range(len(inputs)): # each data synapse to the neuron
				neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
			# the last weight is for the bias
			neuron['weights'][-1] += l_rate * neuron['delta']


# Purpose: Test network weights using a validation test set, return error
# Arguments:
#	  network: system on which to test weights of
#	     data: validation data set
def validate_network(network, data):
	sum_error = 0
	for row in data:
		expected = row[-1] #expected output is last element
		outputs = forward_propagate(network, row)
		out_sum = outputs[0] + outputs[1] + outputs[2]
		outputs = [(i/out_sum) for i in outputs]
		# using sum of squares to determine error
		sum_error += ( 0.5 * sum((expected[i] - outputs[i])**2 
												for i in range(len(expected))))

	total_error = sum_error / len(data)
	return total_error


# Purpose: Train a network for a fixed number of iterations
# Arguments:
#	    train: dataset on which to train
#	   l_rate: learning rate
#	iteration: number of iterations of training
def train_network(network, train, l_rate, iteration):
	error_thresh_reached = 0
	for n in range(iteration):
		sum_error = 0

		for row in train:
			outputs = forward_propagate(network, row)

			# last element of each row is the expected output
			expected = row[-1] 
			
			out_sum = outputs[0] + outputs[1] + outputs[2]
			outputs = [(i/out_sum) for i in outputs]
			
			sum_error += ( 0.5 * sum((expected[i] - outputs[i])**2 
												for i in range(len(expected))))

			backward_propagate_error(network, expected)
			update_weights(network, row, l_rate)

		total_error = sum_error / len(train)
		if total_error < .075:
			error_thresh_reached += 1
			if error_thresh_reached == 5:
				break


# Purpose: Use neural network to identify Iris type
# Arguments:
#	    network: network to use to identify Iris
#	     values: list of 4 float input values 
def identify(network, values):
	output = forward_propagate(network, values)
	index = output.index(max(output))
	if index == 0:
		print("Iris-setosa")
	if index == 1:
		print("Iris-versicolor")
	if index == 2:
		print("Iris-virginica")


#####################################################################
#                   Code to call train/use network                  #
#####################################################################
data = []
labels = []

# Read in data file
f = open("iris.txt", "r")
database = f.readlines()
f.close()


# Clean and categorize data
for line in database:
	line = line.split(",")
	hot = []
	# hot encoding of Iris types
	if line[4] == "Iris-setosa\n":
		hot = [1, 0, 0]
	elif line[4] == "Iris-versicolor\n":
		hot = [0, 1, 0]
	else: # Iris-virginica
		hot = [0, 0, 1]

	line.pop(4)
	line = [float(i) for i in line]
	line.append(hot)
	data.append(line)


#random order indices to decide test set and train set of data
test_ids = np.random.permutation(len(database))

#split data based on permutation
data_train = [data[i] for i in test_ids[:90]]
data_validate = [data[i] for i in test_ids[90:120]]
data_test = [data[i] for i in test_ids[120:150]]


seed(1)
error = 1
learning_rate = 1

# Training network to 92.5% accuracy because 92.5% is the grade I need for an A
network = initialize_network(4, 5, 3)
while error >= 0.075:
	
	#initialize network with random weights
	# 4 input neurons, 5 hidden neurons, 3 output neurons	
	train_network(network, data_train, learning_rate, 500)
	error = validate_network(network, data_validate)
	print("validate error: %.1f%%" % (error * 100))

	if learning_rate < 0.7:
		learning_rate += 0.05
	else:
		learning_rate -= 0.05

error = validate_network(network, data_test)
print("test error: %.1f%%" % (error * 100))