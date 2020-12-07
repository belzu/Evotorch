import os,sys,copy,time
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import warnings
import copy
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd


# ================== BEGIN THE HIERARCHY OF THE CLASSES ==================

# There are two main concepts that compose this library: the descriptors and the implementators

# The descriptors, as their name suggests, describe the attributes that the networks have, such as their
# optimizer, their criterion their input_dim,... etc.
# There is a class called Network_Descriptor which stores all the attributes that are common to all the neural
# networks, no matter their type.
# The other two descriptors are MLP_Descriptor (MultiLayer Perceptron) and CNN_Descriptor (Convolutional Neural Network)
# which store all the attributes that are exclusive to those kind of neural networks.

# On the other hand we can find the implementators, which are the physical embodiment of the descriptors, or, in other words,
# of the neural networks themselves.

# ================== END THE HIERARCHY OF THE CLASSES ==================

# This class stores the main attributes of a network

# lambda a: a 			mini function which returns the argument you passed to it

criterion_funcs = {"nlll":nn.NLLLoss, "mse":nn.MSELoss, "crossentl":nn.CrossEntropyLoss, "bcewll":nn.BCEWithLogitsLoss}
activation_funcs = {"relu":F.relu, "sigmoid":F.sigmoid}#, F.log_softmax, F.softmax}
optimization_funcs = {"adam":optim.Adam, "sgd":optim.SGD}
# ==== BEGIN Descriptors ====
class Network_Descriptor: # In this class we define all the elements than conform the descriptor of a Neural Network
	# This attributes are common to all the neural network descriptors, no matter their type
	# === BEGIN Attributes ===
	# * hidden_fc_layers = contains the list with the number of neurons for each hidden fully connected layer
	# * input_dim = contains the number of input neurons of the very first layer (the number of features of each instance)
	# * output_dim = contains the number of output neurons of the very last layer (the number of classes to classify)
	# * act_functions = contains the activation functions
	# * batch_size = contains the size of the batches
	# * dropout = contains the number corresponding to the dropout of the network
	# * epochs = contains the number of times the entire training set is trained
	# * learning_rate = contains the learning rate of the network
	# * optim_ref = contains the reference to the optimization function
 	# * criter_ref = contains the reference to the criterion
 	# * print_every = defines how often the running loss will be printed
 	# * patience = defines how many times in a row the current running loss can be greater than the previous running loss
	# === END Attributes ===
	def __init__(self, hidden_fc_layers, input_dim, output_dim, act_funcs_ref, dropout, \
		batch_size, epochs, learning_rate, optim_ref, criter_ref, print_every, patience):
		
		self.hidden_fc_layers = hidden_fc_layers
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.act_functions_ref = act_funcs_ref
		self.batch_size = batch_size
		self.dropout = dropout
		self.learning_rate = learning_rate
		self.epochs = epochs
		self.optim_ref = optim_ref
		self.criter_ref = criter_ref
		self.print_every = print_every
		self.patience = patience

	def change_hidden_fc_layers(self, hidden_fc_layers):
		self.hidden_fc_layers = hidden_fc_layers
	def change_input_dim(self,input_dim):
		self.input_dim = input_dim
	def change_output_dim(self,output_dim):
		self.output_dim = output_dim
	def change_act_functions_ref(self,act_functions_ref):
		self.act_functions_ref = act_functions_ref
	def change_batch_size(self,batch_size):
		self.batch_size = batch_size
	def change_dropout(self,dropout):
		self.dropout = dropout
	def change_learning_rate(self, learning_rate):
		self.learning_rate = learning_rate
	def change_epochs(self, epochs):
		self.epochs = epochs
	def change_optim_ref(self, change_optim_ref):
		self.optim_ref = change_optim_ref
	def change_criter_ref(self, change_criter_ref):
		self.criter_ref = change_criter_ref
	def change_print_every(self, print_every):
		self.print_every = print_every
	def change_patience(self, patience):
		self.patience = patience

	# Transforms the data from the txt-native format to Evotorch format
	def transform_format(self, string):
		string = string.replace("[","")
		string = string.replace("]","")
		string = string.replace(" ","")
		string = string.replace("\n","")
		string = string.replace("\"","")
		string = string.replace("'","")
		if "," in string or ',' in string:
			string = string.split(",")
		return string


# We can read class MLP_Descriptor(Network_Descriptor) in the following way:
# class MLP_Descriptor is a type of Network_Descriptor
class MLP_Descriptor(Network_Descriptor):
	# === BEGIN Attributes ===
	# * mode = Defines if the neural network will be in classification mode (0) or regression mode (1)
	# === END Attributes ===
	def __init__(self, hidden_fc_layers=[5,5], input_dim=1, output_dim=1, activation_funcs_ref = "relu_all", dropout = [0.5,0.5],\
		batch_size=1, epochs=1, learning_rate =0.1, optim_ref = "adam", criter_ref =  "nlll", print_every=40, patience = 5,\
		mode="clssf"):
		#when calling this constructor without passing any arguments the values of the parameters are going to be
		#the ones defined when declaring the Network_Descriptor constructor
		super().__init__(hidden_fc_layers, input_dim, output_dim, activation_funcs_ref, dropout, batch_size, epochs,\
			learning_rate, optim_ref, criter_ref, print_every, patience)
		
		self.mode = mode

	def change_mode(self,mode):
		self.mode = mode

	# We save the NN's information on a text file we passed in file_name
	def save_NN_info(self, file_name):
		f = open(file_name, "w")
		info = str(self.hidden_fc_layers)+"\n"+str(self.input_dim)+"\n"+str(self.output_dim)+"\n"+str(self.act_functions_ref)\
		+"\n"+str(self.dropout)+"\n"+str(self.batch_size)+"\n"+str(self.epochs)+"\n"+str(self.learning_rate)+"\n"+\
		str(self.optim_ref)+"\n"+str(self.criter_ref)+"\n"+str(self.print_every)+"\n"+str(self.patience)+"\n"+str(self.mode)
		f.write(info)
		f.close()

	# We load the NN's information from a text file we passed in file_name
	def load_NN_info(self, file_name):
		f = open(file_name, "r")
		hidden_fc_layers_str = self.transform_format(f.readline())
		self.hidden_fc_layers = []
		if type(hidden_fc_layers_str) == list:
			for i in hidden_fc_layers_str:
				self.hidden_fc_layers.append(int(i))
		else:
			self.hidden_fc_layers.append(int(hidden_fc_layers_str))

		self.input_dim = int(f.readline())
		self.output_dim = int(f.readline())
		
		act_functions_refs_str = self.transform_format(f.readline())
		if act_functions_refs_str == "relu_all":
			self.act_functions_ref = "relu_all"
		elif act_functions_refs_str == "sigmoid_all":
			self.act_functions_ref = "sigmoid_all"
		else:
			self.act_functions_ref = []
			if type(act_functions_refs_str) == list:
				for i in act_functions_refs_str:
					self.act_functions_ref.append(i)
			else:
				self.act_functions_ref.append(act_functions_refs_str)
		dropout_refs_str = self.transform_format(f.readline())
		self.dropout = []
		if type(dropout_refs_str) == list:
			for i in dropout_refs_str:
				self.dropout.append(float(i))
		else:
			self.dropout.append(float(dropout_refs_str))

		self.batch_size = int(f.readline())
		self.epochs = int(f.readline())
		self.learning_rate = float(f.readline())

		optim_ref_str = self.transform_format(f.readline())
		self.optim_ref = optim_ref_str

		criter_ref_str = self.transform_format(f.readline())
		self.criter_ref = criter_ref_str

		self.print_every = int(f.readline())
		self.patience = int(f.readline())
		self.mode = self.transform_format(f.readline())
		

class CNN_Descriptor(Network_Descriptor):
	# === BEGIN Attributes ===
	# * conv_layers = contains the list with the number of channels for each convolutional layer
	# * kernel_sizes = defines the sizes of the kernels. It must have these shape: [ [x,y], [x,y],...,[x,y] ]
	# where x defines the kernel size of the i-th convolutional layer and y defines the kernel size of the i-th pooling layer
	# * conv_stride_sizes = defines the sizes of the strides of the convolutional layer. It must have these shape: [x,x,x,...]
	# where x defines the stride size of the i-th convolutional layer
	# === END Attributes ===
	def __init__(self, hidden_fc_layers = [5,5], input_dim = 50, output_dim = 1, \
		activation_funcs_ref = "relu_all", dropout = [0.5,0.5], batch_size = 1, epochs = 1, learning_rate = 0.1,\
		optim_ref = "adam", criter_ref =  "nlll", print_every = 40, patience = 5, conv_layers = [1,32,64], \
		kernel_sizes = [ [5,2],[5,2] ], conv_stride_sizes = [1,1]):	
		
		super().__init__(hidden_fc_layers, input_dim, output_dim, activation_funcs_ref, \
			dropout, batch_size, epochs, learning_rate, optim_ref, criter_ref, print_every, patience)
		
		self.conv_layers = conv_layers # [ number of filters on each convolution layer ]
		self.kernel_sizes = kernel_sizes # [ [kernel size of convolution layer, kernel size of pooling layer] ]
		self.conv_stride_sizes = conv_stride_sizes # [ [stride size of convolution layer, stride size of pooling layer] ]

	def change_conv_layers(self, conv_layers):
		self.conv_layers = conv_layers
	def change_kernel_sizes(self,kernel_sizes):
		self.kernel_sizes = kernel_sizes
	def change_conv_stride_sizes(self, conv_stride_sizes):
		self.conv_stride_sizes = conv_stride_sizes


	# We save the NN's information on a text file we passed in file_name
	def save_NN_info(self, file_name):
		f = open(file_name, "w")
		info = str(self.hidden_fc_layers)+"\n"+str(self.input_dim)+"\n"+str(self.output_dim)+"\n"+\
		str(self.act_functions_ref)+"\n"+str(self.dropout)+"\n"+str(self.batch_size)+"\n"+str(self.epochs)+"\n"+str(self.learning_rate)+"\n"+\
		str(self.optim_ref)+"\n"+str(self.criter_ref)+"\n"+str(self.print_every)+"\n"+str(self.patience)+"\n"+\
		str(self.conv_layers)+"\n"+str(self.kernel_sizes)+"\n"+str(self.conv_stride_sizes)
		f.write(info)
		f.close()


	# We load the NN's information from a text file we passed in file_name
	def load_NN_info(self, file_name):
		f = open(file_name, "r")
		hidden_fc_layers_str = self.transform_format(f.readline())
		self.hidden_fc_layers = []
		if type(hidden_fc_layers_str) == list:
			for i in hidden_fc_layers_str:
				self.hidden_fc_layers.append(int(i))
		else:
			self.hidden_fc_layers.append(int(hidden_fc_layers_str))

		self.input_dim = int(f.readline())
		self.output_dim = int(f.readline())
		
		act_functions_refs_str = self.transform_format(f.readline())
		if act_functions_refs_str == "relu_all":
			self.act_functions_ref = "relu_all"
		elif act_functions_refs_str == "sigmoid_all":
			self.act_functions_ref = "sigmoid_all"
		else:
			self.act_functions_ref = []
			if type(act_functions_refs_str) == list:
				for i in act_functions_refs_str:
					self.act_functions_ref.append(i)
			else:
				self.act_functions_ref.append(act_functions_refs_str)
		dropout_refs_str = self.transform_format(f.readline())
		self.dropout = []
		if type(dropout_refs_str) == list:
			for i in dropout_refs_str:
				self.dropout.append(float(i))
		else:
			self.dropout.append(float(dropout_refs_str))

		self.batch_size = int(f.readline())
		self.epochs = int(f.readline())
		self.learning_rate = float(f.readline())

		optim_ref_str = self.transform_format(f.readline())
		self.optim_ref = optim_ref_str

		criter_ref_str = self.transform_format(f.readline())
		self.criter_ref = criter_ref_str

		self.print_every = int(f.readline())
		self.patience = int(f.readline())


		self.conv_layers = []
		conv_layers_str = self.transform_format(f.readline())
		for i in conv_layers_str:
			self.conv_layers.append(int(i))

		self.kernel_sizes = []
		kernel_sizes_str = self.transform_format(f.readline())
		# As it is a list with this form [ [x,y], [x,y],..., [x,y]] we use the zipping
		kernel_size_pairs = zip(kernel_sizes_str[0::2], kernel_sizes_str[1::2])

		self.kernel_sizes  = [ [int(i),int(j)] for i,j in kernel_size_pairs]

		self.conv_stride_sizes = []

		stride_sizes_str = self.transform_format(f.readline())

		for i in stride_sizes_str:
			self.conv_stride_sizes.append(int(i))
# ==== END Descriptors ====



# ==== BEGIN Implementators ====
class Network(nn.Module):
	# This attributes are common to all the neural networks, no matter their type
	# === BEGIN Attributes ===
	# * descriptor = contains the network descriptor
	# * dropout = contains the dropout layer of the network
	# * act_functions = contains the activation functions
	# * criterion = contains the loss function of the network
	# === END Attributes ===

	def __init__(self,Network_Descriptor):
		
		super().__init__()

		torch.set_default_dtype(torch.float64)

		self.descriptor = Network_Descriptor

		self.dropout = nn.ModuleList([nn.Dropout(p = i) for i in self.descriptor.dropout]) # the length has to be == len(hidden_fc_layers)

		self.act_functions = []

		if self.descriptor.act_functions_ref == "relu_all":
			for i in range(len(self.descriptor.hidden_fc_layers)):
				self.act_functions.append(F.relu)
		elif self.descriptor.act_functions_ref == "sigmoid_all":
			for i in range(len(self.descriptor.hidden_fc_layers)):
				self.act_functions.append(F.sigmoid)
		else:
			for i in self.descriptor.act_functions_ref: # the length of self.descriptor.act_functions_ref has to be == len(hidden_fc_layers)
				self.act_functions.append(activation_funcs[i])

		self.criterion = criterion_funcs[self.descriptor.criter_ref]()#reduction='sum', size_average=False)

	def predict(self, x):
		return self(x)

# Saves the values of the NN's hiperparameters in the file. In other words, it saves the values of the NN's descriptor
	def save_NN_info(self, file_name): 
		self.descriptor.save_NN_info(file_name)

# Saves the NN with all the weights, biases and parameters
	def save_NN(self, file_name): 
		torch.save(self.state_dict(), file_name)

# Loads the NN with all the weights, biases and parameters
	def load_NN(self, file_name, strictt = False):  
		self.load_state_dict(torch.load(file_name), strict=strictt)


class MLP_Network(Network):
	# This attributes are common to all the neural networks, no matter their type
	# === BEGIN Attributes ===
	# * hidden_fc_layers = contains the hidden fully connected layers
	# * output = contains the last layer of the network, the one which computes the prediction
	# * optimizer = contains the optimization algorithm's instance
	# === END Attributes ===

	# The MLPs do not have an initialization different from the ones	 of the generic neural networks
	def __init__(self, network_descriptor):
		super().__init__(network_descriptor)
		self.hidden_fc_layers = nn.ModuleList([nn.Linear(self.descriptor.input_dim, \
			self.descriptor.hidden_fc_layers[0])])

		layer_sizes = zip(self.descriptor.hidden_fc_layers[:-1], self.descriptor.hidden_fc_layers[1:])
		# We add the hidden layers to the neural network
		self.hidden_fc_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
		# We create the last layer within the neural network
		self.output = nn.Linear(self.descriptor.hidden_fc_layers[-1], self.descriptor.output_dim)
		# We initialize the optimizer
		self.optimizer = optimization_funcs[self.descriptor.optim_ref](self.parameters(), \
			lr = self.descriptor.learning_rate)#, weight_decay=0.01)
		# Weight_decay default == 0

# Here we load the hiperparameters from a file we passed as an argument (file_name). 
# file_name must also includes the path regarding the file is in a different location than this very module
	def load_NN_info(self, file_name):
		self.descriptor.load_NN_info(file_name)
		self.__init__(self.descriptor)
	# Here the forward pass is done
	def forward(self, x):
		for linear in range(len(self.hidden_fc_layers)):
			# nn.BatchNorm1d(num_features=320)
			x = self.act_functions[linear](self.hidden_fc_layers[linear](x))
			x = self.dropout[linear](x)
		x = self.output(x) # This is the output layer, so we dont apply the actvation function			
		# When writing self.output(x) we are applying the lineal function in the last layer
		# The output are the probabilities that the input has of belonging to each class
		if self.descriptor.mode  == "clssf":
			return F.log_softmax(x,dim=1) # You can put return F.softmax(x, dim=1)
		else:
			return x

	# THIS IS MY VERSION OF THE TRAINING_MLP, ALONG WITH MY OWN IMPLEMENTATION OF THE EARLY STOPPING
	def training_MLP(self,trainloader):

		# We initialize the variables which will help us detect overfitting
		steps = 0 
		running_loss = 0.0
		halt = False	
		last_loss = float("Inf")
		patience_count = 0.0


		# Writing of data:
		running_losses = []
		epochs = []

		self.optimizer.zero_grad()

		# torch.clamp(input, min, max, out=None)

		# clipping_value = 0.5 # arbitrary value of your choosing

		# print("======== TRAINING PHASE ========\n\n")

		for e in range(self.descriptor.epochs):

			# print("EPOCH ", e)

			for batch in trainloader:

				# We put the optimizers' gradients to zero (important to do this for each batch-iteration)
				self.optimizer.zero_grad()

				steps += 1

				# We extract the values of each element in the batch along with their labels
				values, labels = batch

				# We compute the output by passing doing a forward pass
				output = self.predict(values.view(-1,self.descriptor.input_dim)) # output = self(values.view(-1,self.descriptor.input_dim))
				# We calculate the loss (THE DATA MUST BE IN FLOAT64/double FORMAT, WHICH CAN BE OBTAINED
				# APPLYING THE LONG FUNCTION TO THE TENSORS ==> labels_f64 = labels_f32.long()
				# or labels_f64 = labels_f32.double())

				loss = self.criterion(output,labels)
				# We do the backward-pass
				loss.backward()
				
				# We apply the new gradients to the NN parameters
				self.optimizer.step()

				# This is the loss of the current batch, with which we are going to compute the overfitting
				running_loss = loss.item() 

				# If the running loss has kept rising in self.descriptor.patience number 
				# of steps then we halt the training
				if running_loss > last_loss or running_loss == last_loss:
					patience_count += 1
				else:
					patience_count = 0.0
				if patience_count == self.descriptor.patience:
					halt = True

				# We keep track the the current loss for the next step
				last_loss = copy.copy(running_loss)

				# We print the loss each self.descriptor.print_every number of steps
				# if steps % self.descriptor.print_every == 0:
				# 	if self.descriptor.mode == "rgrss":
				# 		predicted,_  = torch.max(output.data, 1)
				# 		# print("Predicted ", predicted," Real ", labels)
				# 	else:
				# 		_, predicted = torch.max(output.data, 1) 
				# 		# print("Predicted ", predicted," Real ", labels)	
				# 	print("Running loss ", running_loss)	
				
				# If overfitting occurred the batch-loop breaks
				if halt:
					break
			# If overfitting occurred the epoch-loop breaks
			if halt:
				break
		# If overfitting occurred the user gets a notification
			epochs.append(e)
			running_losses.append(running_loss)
		# if halt:
		# 	print("Overfitting occurred!")
		return epochs, running_losses

	# IF WE WANT MULTILABEL CLASSIFICATION THIS IS THE FORMULA
	# output = criterion(values,labels)
	# output = torch.sigmoid(output)  # torch.Size([N, C]) e.g. tensor([[0., 0.5, 0.]])
	# output[output >= 0.5] = 1
	# accuracy = (output == labels).sum()#/(N*C)*100 # i understand that (N*C) == total (?)
	# accuracy = (output == labels).sum()#/total*100 

	def testing_MLP(self,testloader,_print_=False, save=False, pic_name="a"):
		correct = 0
		total = 0
		summ = 0
		summ2 = 0
		result = -1

		plt.clf()
		plt.xlim(-1.0, 1.0)
		plt.ylim(-1.0, 1.0)
		
		# print("======== TESTING PHASE ========\n\n")
		predicted_data = []
		real_data = []

		# Variables that will help as get the regression's result implementing MSE with Pytorch's implementation
		outputl_MSE = []
		labell_MSE = []
		for i, (inputs,labels) in enumerate(testloader): 
			# We get the values and the labels of each batch
			# We get the number of features for each instance on each batch	
			output = self.predict(inputs.view(-1,self.descriptor.input_dim))#output = self(inputs.view(-1,self.descriptor.input_dim))
			# print("Output ", output)		

			if self.descriptor.mode == "clssf":
				# If the neuron with highest value (highest probability) corresponds to the
				# neuron with the index equal to the value of the label then
				# it will be considered a correct classification					
				
				# We get the index with the maximum value (the neuron with the highest probability)
				# We get the predicted class (classification)
				# This is the way Pytorch implements the accuracy calculation

				_, predicted = torch.max(output.data, 1) # We get the index
				total += labels.size(0)
				# print("Predicted ", predicted, "labels ", labels)
				correct += (predicted == labels).sum().item()
				# print("Corrects: ", correct)


			#elif self.descriptor.mode == Multilabel classification:
				# total += labels.size(0)
				# output = torch.sigmoid(output)  # torch.Size([N, C]) e.g. tensor([[0., 0.5, 0.]])
				# output[output >= 0.5] = 1
				# correct += (output == labels).sum()

			else:
				predicted,_  = torch.max(output.data, 1) # predicted = copy.copy(output)  # We get the value
				total += labels.size(0)
				# print("Predicted ", predicted, "labels ", labels)
				# We are doing sum( (predicted[i]-real[i])^2 ) 0<=i<=number of testing instances
				for i in range(len(predicted)):
					# We record the predictions and the real data in order to put them in a graph
					predicted_data.append(predicted[i].item())
					real_data.append(labels[i].item())
					# Second way of calculating the MSE (Doing it using nn.MSELoss)
					# We record the outputs and their corresponding real values
					outputl_MSE.append(predicted[i].item())
					labell_MSE.append(labels[i].item())

		# Here we print the result of the regression 
		if _print_ and self.descriptor.mode == "rgrss":		
			# plt.legend(loc='best')
			plt.title("Prediction")
			# plt.scatter(torch.max(i).item(),labels[idx].item(),label="Result",alpha=0.5)
			plt.scatter(real_data, predicted_data)
			plt.xlabel("Original data")
			plt.ylabel("Predicted data")
			plt.legend()		
			plt.show()

		if save and self.descriptor.mode == "rgrss":		
			# plt.legend(loc='best')
			plt.title("Prediction")
			# plt.scatter(torch.max(i).item(),labels[idx].item(),label="Result",alpha=0.5)
			plt.scatter(real_data, predicted_data)
			plt.xlabel("Original data")
			plt.ylabel("Predicted data")
			plt.legend()
			plt.savefig(pic_name)


		# The users gets notificated about the accuracy of the classification / regression
		if self.descriptor.mode == "clssf": # or self.descriptor.mode == Multilabel classification
			# print("Total ", total)
			result = round(correct/total,3) # We get the accuracy
			# print(correct," correct out of ", total)
			# print("Accuracy: ", result)
		else:			
			# Second way of calculating the MSE (Doing it using nn.MSELoss)
			# We pass the lists to Tensors and compute the loss first instanciating a nn.MSELoss algorithm object
			# and then calculating the loss using it
			outputl_MSE = torch.Tensor(outputl_MSE)
			labell_MSE = torch.Tensor(labell_MSE)

			MSE_calc = nn.MSELoss()

			result = MSE_calc(outputl_MSE, labell_MSE).item()

			# print("Mean loss using MSELoss ", result)
			# print("Mean loss using nn.MSELoss ", result2)
		# The accuracy is returned
		return result#accur


class CNN_Network(Network):
	# === BEGIN Attributes ===
	# * conv_layers = contains the sequences of (convolutional layers + relu layers + MaxPooling layers)
	# * _to_linear = contains the number of neurons that the hidden fc layer connected to the las conv_layer has
	# * hidden_fc_layers = contains the hidden fully connected layers
	# * output = contains the last layer of the network, the one which computes the prediction
	# * optimizer = contains the optimization algorithm's instance
	# === END Attributes ===
	def __init__(self,network_descriptor):

		super().__init__(network_descriptor)

		layer_sizes = zip(self.descriptor.conv_layers[:-1], self.descriptor.conv_layers[1:])
		# When we define a convolutional layer what we are really defining is a sequence of three layers:
		# Convolutional layer + Relu layer + MaxPooling layer
		
		self.conv_layers = nn.ModuleList([nn.Sequential(
			nn.Conv2d(h1, h2, kernel_size = self.descriptor.kernel_sizes[i][0], stride = self.descriptor.conv_stride_sizes[i], \
				padding = 0),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = self.descriptor.kernel_sizes[i][1])) for i,(h1, h2) in enumerate(layer_sizes) ] )

		# # This random value is created in order to get the self._to_linear value, which is very useful
		x = torch.randn(self.descriptor.input_dim, self.descriptor.input_dim).view(-1,1, \
			self.descriptor.input_dim,self.descriptor.input_dim)
		# This variable transforms the data we have passed through the convolution function so that we can
		# forward-pass it
		self._to_linear = None
		# # Here we get the value of self._to_linear, which we will use in order to shape the size of the data correctly
		try:
			self.convs(x)
			# a = self.convs_sizes(x.shape[3])
			# print(r.shape[3], a)
			# size_is_1 = False
			# numb_of_convl = len(self.descriptor.kernel_sizes)
			# i = 0
			# _size_ = list(x.size())[3]
			# while not size_is_1 and i < numb_of_convl:
			# 	_size_ = self.get_tensor_sz_after_convpool(list(x.size())[3])
			# 	print(i, numb_of_convl)
			# 	if _size_ <= 1 or self.descriptor.kernel_sizes[i][0]>=_size_ or self.descriptor.kernel_sizes[i][1]>=_size_:
			# 		size_is_1 = True
			# 		self.conv_layers = self.conv_layers[:i+1]
			# 		self.descriptor.conv_layers = self.descriptor.conv_layers[:i+1]
			# 		self.descriptor.kernel_sizes = self.descriptor.kernel_sizes[:i]
			# 		self.descriptor.conv_stride_sizes = self.descriptor.conv_stride_sizes[:i]
			# 		numb_of_convl = len(self.descriptor.conv_layers)
			# 	else:
			# 		i+=1
			# self._to_linear = x.shape[1]*x.shape[2]*self.convs_sizes(list(x.size())[3])
		except Exception as e:
			raise Exception("Convolution failed!")


		# Above, the process is the same as in MLP_Network and fills the same purpose
		self.hidden_fc_layers = nn.ModuleList([nn.Linear(self._to_linear, self.descriptor.hidden_fc_layers[0])])

		layer_sizes = zip( self.descriptor.hidden_fc_layers[:-1], self.descriptor.hidden_fc_layers[1:] )

		self.hidden_fc_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

		self.output = nn.Linear(self.descriptor.hidden_fc_layers[-1], self.descriptor.output_dim)
		# We initialize the optimizer

		self.optimizer = optimization_funcs[self.descriptor.optim_ref](self.parameters(), \
			lr = self.descriptor.learning_rate)#, weight_decay=0.01)
		# print("CNN initialization successful")

# Here we load the hiperparameters from a file we passed as an argument (file_name). 
# file_name also includes the path
	def load_NN_info(self, file_name): 
		self.descriptor.load_NN_info(file_name)
		self.__init__(self.descriptor)

	def get_conv_output_size(self, x_size, conv_kernel_size, conv_stride_size):
		# ((W1−F  +2P)/S)  +1
		# [(I – F +2 *P) / S] +1 x D | I: input , F: filter size, P: pooling, D: number of feature maps
		# P == pooling == 0
		return ((x_size-conv_kernel_size)/conv_stride_size)+1 

	def get_pool_output_size(self, x_size, pool_kernel_size):
		# (W1−F)/S  +1
		# # [(I – F) / S] + 1 x D | I: input , F: filter size, D: number of feature maps
		# pool_kernel_size == pool_stride_size	
		return ((x_size-pool_kernel_size)/pool_kernel_size)+1

	def get_convpool_size(self, x_size, conv_kernel_size, conv_stride_size, pool_kernel_size):
		x_sz = self.get_conv_output_size(x_size, conv_kernel_size, conv_stride_size)
		x_sz = self.get_pool_output_size(x_sz, pool_kernel_size)
		return x_sz


	def get_tensor_sz_after_convpool(self, x_size):
		for i in range(len(self.descriptor.kernel_sizes)):
			x_size = int(self.get_convpool_size(x_size, self.descriptor.kernel_sizes[i][0],\
				self.descriptor.conv_stride_sizes[i], self.descriptor.kernel_sizes[i][1]))
		return x_size




	def convs_sizes(self, x):
		original_x = copy.deepcopy(x)
		# print("Estos son los conv layers ", self.conv_layers)
		for i,conv in enumerate(self.conv_layers):
			x = self.get_tensor_sz_after_convpool(x)
			# print( "Size of the convolution ", self.get_conv_output_size(x) ,list(x.size()))
			if x <= 1 or self.descriptor.kernel_sizes[i][0]>=x or self.descriptor.kernel_sizes[i][1]>=x:
				# print("Da tamaino 1 con conv_layers ", self.descriptor.conv_layers)
				self.conv_layers = self.conv_layers[:-1]
				self.descriptor.conv_layers = self.descriptor.conv_layers[:-1]
				self.descriptor.kernel_sizes = self.descriptor.kernel_sizes[:-1]
				self.descriptor.conv_stride_sizes = self.descriptor.conv_stride_sizes[:-1]
				x = self.convs_sizes(original_x)
				break
		return x



# In this function the convolution and the pooling take place
	def convs(self, x):
		original_x = copy.deepcopy(x)
		for i,conv in enumerate(self.conv_layers):
			x = conv(x)
			_size_ = list(x.size())[3]
			if _size_ <= 1 or self.descriptor.kernel_sizes[i][0]>=_size_ or self.descriptor.kernel_sizes[i][1]>=_size_:
				# print("Da tamaino 1 con conv_layers ", self.descriptor.conv_layers)
				self.conv_layers = self.conv_layers[:-1]
				self.descriptor.conv_layers = self.descriptor.conv_layers[:-1]
				self.descriptor.kernel_sizes = self.descriptor.kernel_sizes[:-1]
				self.descriptor.conv_stride_sizes = self.descriptor.conv_stride_sizes[:-1]
				x = self.convs(original_x)
				break
		if self._to_linear is None:
			self._to_linear = x.shape[1]*x.shape[2]*x.shape[3]
		return x


# Note that tensor.shape is an alias to tensor.size(), though tensor.shape is an
# attribute of the tensor in question whereas tensor.size() is a function.

	def forward(self, x):
		# We make the convolution
		x = self.convs(x)
		x = x.reshape(x.size(0), -1) # This line transforms x's dimensionality into: [batch_size, self._to_linear]
		# # We put the data on its corresponding format (I THINK IT IS DEPRECATED)
		# x = x.view(-1,1, self._to_linear)
		# We do the forward pass
		for linear in range(len(self.hidden_fc_layers)):
			x = self.act_functions[linear](self.hidden_fc_layers[linear](x))
			x = self.dropout[linear](x)
		x = self.output(x) # This is the output layer, so we dont apply the actvation function	
		return F.log_softmax(x, dim=1)



	def training_CNN(self,training_data):
		
		steps = 0 
		running_loss = 0.0
		patience_count = 0
		halt = False
		last_loss = float("Inf")
		running_losses = []
		epochs = []
		self.optimizer.zero_grad()

		# print("======== TRAINING PHASE ========\n\n")

		for e in range(self.descriptor.epochs):
			
			# print("EPOCH ", e)

			for batch in training_data:
				values, labels = batch
				# We will have to delete the following two lines
				# values = values.double()
				# labels = labels.long()

				values = values.view(-1, 1, self.descriptor.input_dim, self.descriptor.input_dim)
				# We put the gradients to zero
				self.optimizer.zero_grad()
				# We update the number of steps		
				steps+=1

				output = self.predict(values.double())#output = self(values.double())
				# We transform the dimensions of the output
				output = output.view(-1,self.descriptor.output_dim)
				# We compute the loss
				if list(output.size())[0] != list(labels.size())[0]:
					output = output.view(self.descriptor.batch_size, -1)

				loss = self.criterion(output, labels)
				# We do the backpropagation
				loss.backward()
				# We apply the changes
				self.optimizer.step()
				# We obtain the running loss (the actual loss)
				running_loss = loss.item() # This is the loss of the current batch, with which we are going to compute the overfitting

				# If the running loss has kept rising in self.descriptor.patience number of steps then
				# we halt the training
				if running_loss > last_loss or running_loss == last_loss:
					patience_count += 1
				else:
					patience_count = 0.0
				# If the patience is reached, the training is halted
				if patience_count == self.descriptor.patience:
					halt = True
				# We keep record of the running loss for the next step
				last_loss = copy.copy(running_loss)

				# The code within this is statement has the purpose of printing the running loss
				# We print the running loss
				# if steps % self.descriptor.print_every == 0:	
				# 	print("Running loss ", running_loss)
				# If the patience is reached, the training is halted
				if halt:
					break			
			# If the patience is reached, the training is halted
			if halt:
				break
			epochs.append(e)
			running_losses.append(running_loss)
		# if halt:
		# 	print("Overfitting occurred!")
		return epochs, running_losses
	# def training_CNN2(self, testloader):


	def testing_CNN(self,testloader):
		# We initialize the test loss
		test_loss = 0
		# We initialize the number of correct predictions
		correct = 0
		# We initialize the number of total predictions
		total = 0

		# print("======== TESTING PHASE ========\n\n")
		for batch in testloader:
		# for i in tqdm(range(0, len(testing_data), self.descriptor.batch_size)): (deprecated)
			# We get the values and the labels
			values, labels = batch
			
			# We will have to delete the following two lines
			# values = values.double()
			# labels = labels.long()
			values = values.view(-1, 1, self.descriptor.input_dim, self.descriptor.input_dim)

			output = self.predict(values.double())#output = self(values.double())
			# We redimension the output
			output = output.view(-1,self.descriptor.output_dim)

			if list(output.size())[0] != list(labels.size())[0]:
				output = output.view(self.descriptor.batch_size, -1)
			# We get the predictions
			_, predicted = torch.max(output.data, 1) # We get the index
			# print("Predicted ", predicted, "labels ", labels)
			# We get the total number of instances
			total += labels.size(0)
			# We get the how many predictions were correct
			correct += (predicted == labels).sum().item()
			# print("Corrects: ", correct)

		# we get, print and return the accuracy
		result = round(correct/total,3) # We get the accuracy
		# print(correct," correct out of ", total)
		# print("Accuracy: ", result)
		return result
# ==== END Implementators ====


# What we are doing here is to call MLP_Descriptor() and thus pass an instance of the class MLP_Descriptor to MLP constructor
# which will store in its descriptor attribute that instance of MLP_Descriptor
# MLP_network = MLP(MLP_Descriptor())