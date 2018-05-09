#!/usr/bin/python
import numpy as np

def sigmoid(x,deriv):
	if(deriv == True):
		return x * (1-x)
	else:
		return 1/(1-numpy.exp(-x))


data = []
labels = []

f = open("iris.txt", "r")
database = f.readlines()
f.close()

for line in database:
	line = line.split(",")
	if line[4] == "Iris-setosa\n":
		labels.append([1,0,0])
	elif line[4] == "Iris-versicolor\n":
		labels.append([0,1,0])
	else: # Iris-virginica
		labels.append([0,0,1])

	data.append([float(i) for i in line[:4]])

np.random.seed(1)

#random order indices to decide test set and train set of data
test_ids = np.random.permutation(len(data))

#split data based on permutation
data_train = np.array([data[i] for i in test_ids[:90]])
data_validate = np.array([data[i] for i in test_ids[90:120]])
data_test = np.array([data[i] for i in test_ids[120:150]])

labels_train = np.array([labels[i] for i in test_ids[:90]])
labels_validate = np.array([labels[i] for i in test_ids[90:120]])
labels_test = np.array([labels[i] for i in test_ids[120:150]])

num_examples = len(data_train)

nn_input_dim = 4
nn_output_dim = 3

# Gradient descent parameters 
epsilon = 0.01 # learning rate for gradient descent
reg_lambda = 0.01 # regularization strength


def calculate_loss(model):
	W1, biase1, W2, biase2 = model['W1'], model['biase1'], model['W2'], model['biase2']

	# forward propagation time
	z1 = data_train.dot(W1) + biase1 # input of layer 1
	a1 = np.tanh(z1) # output of layer 1
	z2 = a1.dot(W2) + biase2
	exp_scores = np.exp(z2)
	probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
	# calculate loss
	total = 0
	for x in range(0,int(probs.size/3)):
		total += (labels_train[x][0] * np.log(probs[x][0]))
		total += (labels_train[x][1] * np.log(probs[x][1]))
		total += (labels_train[x][2] * np.log(probs[x][2]))
	data_loss = total/probs.size * -1
	#correct_logprobs = -np.log(probs[range(num_examples), [np.argmax(output) for output in labels_train]])
	return data_loss

def predict(model, x):
	W1, biase1, W2, biase2 = model['W1'], model['biase1'], model['W2'], model['biase2']

	#Forward propagation
	z1 = x.dot(W1) + biase1
	a1 = np.tanh(z1)
	z2 = a1.dot(W2) + biase2
	exp_scores = np.exp(z2)
	probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
	return np.argmax(probs, axis=1)


def build_model(nn_hdim, num_passes=20000, print_loss=False):
	np.random.seed(0)
	W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
	biase1 = np.zeros((1,nn_hdim))
	W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
	biase2 = np.zeros((1, nn_output_dim))
	model = {}

	for i in range(0, num_passes):

		#forward propagate

		z1 = data_train.dot(W1) + biase1
		a1 = np.tanh(z1)
		z2 = a1.dot(W2) + biase2
		exp_scores = np.exp(z2)
		probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

		#backpropagation
		delta3 = probs
		for j in range(0 , int(delta3.size/3)):
			index = np.argmax(labels_train[j])
			delta3[j][index] -= 1

		dW2 = (a1.T).dot(delta3)
		db2 = np.sum(delta3, axis=0, keepdims=True)
		delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
		dW1 = np.dot(data_train.T, delta2)
		db1 = np.sum(delta2, axis=0)

		dW2 += reg_lambda * W2
		dW1 += reg_lambda * W1

		W1 += -epsilon * dW1
		biase1 += -epsilon * db1
		W2 += -epsilon * dW2
		biase2 += -epsilon * db2

		model = {'W1': W1, 'biase1': biase1, 'W2': W2, 'biase2': biase2}

		if print_loss and i % 1000 == 0:
			print('Loss after iteration {}: {}'.format(i, calculate_loss(model)))
	return model


build_model(5,200000,True)