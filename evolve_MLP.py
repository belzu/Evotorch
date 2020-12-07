import warnings,os,sys
warnings.simplefilter('ignore')
import random,time,copy
from deap import base
from deap import creator
from deap import tools
import numpy as np
import pandas as pd
import torch
from itertools import repeat
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from Network import MLP_Network
from Network import MLP_Descriptor
from GenAlgs import GenAlg_MLP
from GenAlgs import GenAlg_MLP_Descriptor
from sklearn.preprocessing import MinMaxScaler

# ===== HOW TO EXECUTE THIS PROGRAM =====
# python exec_MLP.py <number of layers> <Dataset> <mode>


def read_NN_data(file, mode, batch_size, shuffle, normalize):

	if file[-3:] != "csv":
		print("The file " + file + " is not recognized as a csv extension type file")
		print("Treat it like it was " + file+".csv ? \t [Yes -> 1 | No -> 0]")
		resp = int(input())
		while(resp != 0 and resp != 1):
			print("Treat it like it was " + file+".csv ? \t [Yes -> 1 | No -> 0]")
			resp = int(input())
			print(type(resp))
		if resp == 1:
			file+=".csv"
	try:
		data_pd = pd.read_csv(file,sep='\\t')
		data = torch.from_numpy(np.array(data_pd.values, dtype='float64'))

		data_np = np.array(data_pd.values, dtype='float64')
		# data = np.array(data_pd.values,dtype='float64')
	except:
		data_pd = pd.read_csv(file,sep=',')
		# data = np.array(data_pd.values,dtype='float64')
		data = torch.from_numpy(np.array(data_pd.values, dtype='float64'))

		data_np = np.array(data_pd.values, dtype='float64')
	
	num_of_inputs = data_np.shape[1]-1#data.size()[1]-1
	# print('Numero de instancias: ', data.size()[0],'\nNumero de caracteristicas: ', num_of_inputs)
	print('Numero de instancias: ', data_np.shape[0],'\nNumero de caracteristicas: ', num_of_inputs)
	has_class_cero = False
	output_classes = []
	for instance in data_np:
		if not np.isnan(instance[-1].item()):
			if int(instance[-1].item()) not in output_classes:
				output_classes.append(int(instance[-1].item()))
				if int(instance[-1].item()) == 0:
					has_class_cero = True
	

	print("Begins with class cero: ", has_class_cero)

	if mode == "clssf":
		while not has_class_cero:
			for instance in data_np:
				if not np.isnan(instance[-1].item()):
					instance[-1] -= 1
					if instance[-1] == 0:
						has_class_cero = True

	num_of_training = int(int(data.size()[0])*(2/3))

	mod_num_of_training = int((int(data.size()[0])*2) % 3) + (2*int(int(data.size()[0])%6))

	num_of_validation = num_of_training+mod_num_of_training+int(int(data.size()[0])*(1/6))

	standarized_trainloader = []
	for i in data_np[:num_of_training+mod_num_of_training]:
		nan = False
		for j in i[:]:
			if np.isnan(j.item()):
				nan = True
		if nan:
			continue
		else:
			if mode == "clssf":
				standarized_trainloader.append([torch.from_numpy(np.array(i[:-1])).double(), \
					torch.from_numpy(np.array(i[-1])).long()])
			else:
				standarized_trainloader.append(i)
	
	standarized_validloader = []
	for i in data_np[num_of_training+mod_num_of_training:num_of_validation]:
		nan = False
		for j in i[:]:
			if np.isnan(j.item()):
				nan = True
		if nan:
			continue
		else:
			if mode == "clssf":
				standarized_validloader.append([torch.from_numpy(np.array(i[:-1])).double(), \
					torch.from_numpy(np.array(i[-1])).long()])
			else:
				standarized_validloader.append(i)


	standarized_testloader = []

	for i in data_np[num_of_validation:]:
		nan = False
		for j in i[:]:
			if np.isnan(j.item()):
				nan = True
		if nan:
			continue
		else:
			if mode == "clssf":
				standarized_testloader.append([torch.from_numpy(np.array(i[:-1])).double(), \
					torch.from_numpy(np.array(i[-1])).long()])
			else:
				standarized_testloader.append(i)


	if normalize:
		scaler = MinMaxScaler()
		scaler.fit(standarized_trainloader)
		standarized_trainloaderr = scaler.transform(standarized_trainloader)

		scaler.fit(standarized_validloader)
		standarized_validloaderr = scaler.transform(standarized_validloader)

		scaler.fit(standarized_testloader)
		standarized_testloaderr = scaler.transform(standarized_testloader)

		train = [ [torch.from_numpy(np.array(i[:-1])).double(), torch.from_numpy(np.array(i[-1])).double()] \
			for i in standarized_trainloaderr]

		valid = [ [torch.from_numpy(np.array(i[:-1])).double(), torch.from_numpy(np.array(i[-1])).double()] \
			for i in standarized_validloaderr]

		test = [ [torch.from_numpy(np.array(i[:-1])).double(), torch.from_numpy(np.array(i[-1])).double()] \
			for i in standarized_testloaderr]

	else:
		train = copy.copy(standarized_trainloader)

		valid = copy.copy(standarized_validloader)

		test = copy.copy(standarized_testloader)


	trainloader = torch.utils.data.DataLoader(train, batch_size= int(batch_size), \
		shuffle = shuffle, drop_last=True)
	validloader = torch.utils.data.DataLoader(valid, batch_size= int(batch_size), \
		shuffle = shuffle, drop_last=True)
	testloader = torch.utils.data.DataLoader(test, batch_size= int(batch_size),\
		shuffle = shuffle, drop_last=True)
	

	if mode == "clssf":
		num_of_outputs = len(output_classes)
	else:
		num_of_outputs = 1

	return trainloader, validloader, testloader, num_of_inputs, num_of_outputs

def toLongFormat(x):
	for i in x:
		print(type(i.item()))
		i = i.double()
		print(type(i.item()))
	return x
	

def add_instance():
	x = torch.randn(10)#.double() # Converts float 32 to float 64
	# x = toLongFormat(x)
	x = x.double()
	out = 0
	for i in x:
		out += i
	out = out.view(1)
	# x = torch.cat((x, out), -1)
	return [x,out]


def read_NN_data2(batch_size, shuffle):
	standarized_trainloader = []
	standarized_validloader = []
	standarized_testloader = []

	for i in range(1000):
		standarized_trainloader.append(add_instance())
	for i in range(100):
		standarized_validloader.append(add_instance())
		standarized_testloader.append(add_instance())

	trainloader = torch.utils.data.DataLoader(standarized_trainloader, batch_size= int(batch_size), \
		shuffle = shuffle, drop_last=True)
	validloader = torch.utils.data.DataLoader(standarized_validloader, batch_size= int(batch_size), \
		shuffle = shuffle, drop_last=True)
	testloader = torch.utils.data.DataLoader(standarized_testloader, batch_size= int(batch_size),\
		shuffle = shuffle, drop_last=True) # PUT THIS IN THE ORIGINAL
	return trainloader, validloader, testloader, 10, 1


# === for data reading ===
file = sys.argv[1]
mode = int(sys.argv[2])
if mode == 0:
	mode = "clssf"
else:
	mode = "rgrss"

batch_size = 1#64#1
shuffle = True

# === for the genetic algorithm ===
toolbox = base.Toolbox

neuron_min_lim = 1
neuron_max_lim = 50

lr_max_lim = 0.001
lr_min_lim = 0.00001

epochs_min_lim = 10
epochs_max_lim = 30

patience_min_lim = 7
patience_max_lim = 10

pop_numb = 1#50#5

mutation_prob = 0.4
mutation_happen_prob = 0.5
mutate_all_hparams = False
mate_happen_prob = 0.3
mate_all_hparams = False
tournsize = 3
generation_numb = 1#10#50

fc_layer_numb_min_lim = 1
fc_layer_numb_max_lim = 2

dropout_min_lim = 0.4
dropout_max_lim = 0.9

# 0: hddn_fc_lyrs # 1 lr # 2 epochs # 3 patience
# evol_hparameters = ["hddn_fc_lyrs","lr","epcs","pat","hddn_fc_lyrs_sz"]  #"hddn_fc_lyrs_sz" "act_fncs", #drpt
evol_hparameters = ["hddn_fc_lyrs", "lr","epcs","pat","hddn_fc_lyrs_sz","act_fncs", "drpt"]

if mode == "clssf":
	normalize = False # MODE 0
	learning_rate = 0.00001
else:
	normalize = True # MODE 1
	learning_rate = 0.001



hidden_fc_layers = [40,40] #THIS MUST BE DEFINED BECAUSE OF THE LENGTH (EVEN ITS CONTENT IS JUST None-s)
epochs = 2
dropout = [0.5, 0.5]
print_every = 20
patience = 10
act_functions = "relu_all"

# a = F.relu

# dicc = {0:F.relu}

# print(dicc[a])
# exit()

# optim_ref = 0 # MODE 0
# optim_ref = 1 # MODE 1
# criter_ref = 0 # MODE 0
# criter_ref = 1 # MODE 1

if mode == "rgrss":
	optim_ref = "sgd"
	criter_ref = "mse"
	wanted_fitness = 0.001
	# trainloader,validloader,testloader,num_of_inputs,num_of_outputs = read_NN_data2(\
	# 	batch_size, shuffle)
	# trainloader,validloader,testloader,num_of_inputs,num_of_outputs = read_NN_data(file, mode, batch_size, \
	# shuffle, normalize)
else:
	optim_ref = "adam"
	criter_ref = "nlll"
	wanted_fitness = None#0.95

trainloader,validloader,testloader,num_of_inputs,num_of_outputs = read_NN_data(file, mode, batch_size, \
	shuffle, normalize)


# r = [0 for i in range(8)]
# for batch in trainloader:
# 	_,labels = batch
# 	r[int(labels)]+=1

# print(r)
# exit()


# trainloader,validloader,testloader,num_of_inputs,num_of_outputs = read_NN_data2(\
	# 	batch_size, shuffle)

genetic_algorithm = GenAlg_MLP(GenAlg_MLP_Descriptor(toolbox, neuron_min_lim, neuron_max_lim, lr_min_lim, lr_max_lim, epochs_min_lim, epochs_max_lim,\
	patience_min_lim, patience_max_lim, fc_layer_numb_min_lim, fc_layer_numb_max_lim, dropout_min_lim, dropout_max_lim, \
	pop_numb, mutation_prob, mutation_happen_prob, mutate_all_hparams, mate_happen_prob, mate_all_hparams, tournsize, generation_numb, evol_hparameters, wanted_fitness,\
	hidden_fc_layers, num_of_inputs, num_of_outputs, act_functions, dropout, batch_size, epochs, learning_rate, optim_ref, criter_ref, print_every, patience, mode))

start_time = time.time()

genetic_algorithm.simple_genetic_algorithm(trainloader,validloader)

print("Execution time: ", time.time()-start_time, " seconds | ", (time.time()-start_time)/60, " minutes | ",  (time.time()-start_time)/3600, " hours")