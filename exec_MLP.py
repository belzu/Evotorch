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
from statistics import stdev as stdev
from statistics import mean as mean
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
		i = i.long()
	return x

def add_instance():
	x = torch.randn(10).double() # Converts float 32 to float 64
	out = 0
	for i in x:
		out += i
	out = out
	out = out.view(1)
	# x = torch.cat((x, out), -1)
	return [x,out]


def read_NN_data2(batch_size, shuffle):
	standarized_trainloader = []
	standarized_validloader = []
	standarized_testloader = []

	for i in range(10000):
		standarized_trainloader.append(add_instance())
	for i in range(200):
		standarized_validloader.append(add_instance())
		standarized_testloader.append(add_instance())

	trainloader = torch.utils.data.DataLoader(standarized_trainloader, batch_size= int(batch_size), \
		shuffle = shuffle, drop_last=True)
	validloader = torch.utils.data.DataLoader(standarized_validloader, batch_size= int(batch_size), \
		shuffle = shuffle, drop_last=True)
	testloader = torch.utils.data.DataLoader(standarized_testloader, batch_size= int(batch_size),\
		shuffle = shuffle, drop_last=True) # PUT THIS IN THE ORIGINAL
	return trainloader, validloader, testloader, 10, 1


file = sys.argv[1]
shuffle = True

# Interesting for the future
# a = F.relu
# dicc = {F.relu:0,F.sigmoid:1}

# print(dicc[a])
# print(list(dicc.keys())[list(dicc.values()).index(1)])
# exit()


n_network = MLP_Network(MLP_Descriptor())

# Introduce here the txt file with the NN's information (must have the correct format)

#=== BEGIN MODE 0 ===
# NN_info_file = "Arquitecturas/NN_mushroom_muy_bien.txt"
# NN_info_file = "Arquitecturas/NN_forest_type.txt"
# NN_info_file = "Arquitecturas/NN_dermatology_muy_bien.txt"
# NN_info_file = "Arquitecturas/NN_dermatology_0929.txt"
# NN_info_file = "Arquitecturas/NN_audit_risk_bien.txt" 
#=== END MODE 0 ===


#=== BEGIN MODE 1 ===
# NN_info_file = "Arquitecturas/NN_Istambul_predic.txt"
# NN_info_file = "Arquitecturas/NN_forest_simplified_no_tan_mal.txt"
# NN_info_file = "Arquitecturas/NN_forest_fire_bien.txt"
# NN_info_file = "Arquitecturas/NN_bikes_hours_bien.txt"
NN_info_file = "NN_25-08-2020_22_23_06.txt"
#=== END MODE 1 ===

n_network.load_NN_info(NN_info_file)

# if normalization technique uses MAX and MIN then a == MAX and b == MIN
# if normalization technique uses MEAN and STDEV then a == MEAN and b == STDEV
# a and b are used to denormalize the data in orden to be able to see it on its original format

if n_network.descriptor.mode == "clssf":
	normalize = False#True
	# trainloader,validloader,testloader,num_of_inputs,num_of_outputs = read_NN_data(file, \
	# 	n_network.descriptor.mode, n_network.descriptor.batch_size, shuffle, normalize)

else:
	normalize = True
	# trainloader,validloader,testloader,num_of_inputs,num_of_outputs = read_NN_data2(\
	# n_network.descriptor.batch_size, shuffle)

trainloader,validloader,testloader,num_of_inputs,num_of_outputs = read_NN_data(file, \
	n_network.descriptor.mode, n_network.descriptor.batch_size, shuffle, normalize)



start_time = time.time()

# [ [ i[:-1] ,i[-1] ] ]

# === BEGIN Dummy example of MULTILABEL CLASSIFICATION ===
# x = torch.randn(3)
# y = np.array([0.0,1.0])  # works
# y = torch.from_numpy(y)

# Tensor of integer numbers to one hot encoding transformation
# labels = torch.tensor([1, 4, 1, 0, 5, 2])
# labels = labels.unsqueeze(0)
# target = torch.zeros(labels.size(0), 15).scatter_(1, labels, 1.)
# target = target.squeeze(0)
# print(target)
# # exit()

# falso_trainloader = [ [x,target.double()] ]

# f_trainloader = torch.utils.data.DataLoader(falso_trainloader, batch_size= 1, \
# 		shuffle = True)

print(n_network.descriptor.input_dim)
n_network.training_MLP(trainloader)
print("Accuracy del train ", n_network.testing_MLP(validloader))

# Here we test the data
with torch.no_grad():
	if n_network.descriptor.mode == "clssf":
		print("The testing gives an accuracy of ", str( round(100.0 * n_network.testing_MLP(testloader,True), 3 ) ) + " %")
	else:
		print("The testing gives a mean loss of  ", str(n_network.testing_MLP(testloader,True) ) )
	print("Execution time: ", time.time()-start_time, " seconds | ", (time.time()-start_time)/60, " minutes | ",  \
		(time.time()-start_time)/3600, " hours")