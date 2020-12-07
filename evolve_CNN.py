import warnings,os,sys
warnings.simplefilter('ignore')
import random,time,copy
from deap import base
from deap import creator
from deap import tools
from itertools import repeat
from Network import CNN_Network
from Network import CNN_Descriptor
from GenAlgs import GenAlg_CNN
from GenAlgs import GenAlg_CNN_Descriptor
import torch
import torchvision
from torchvision import transforms, datasets
import copy
import cv2
import numpy as np
from tqdm import tqdm
import json as json
from sklearn.datasets import fetch_openml
from torch.utils.data.sampler import SubsetRandomSampler







# df.to_csv("./Results_evolutions/{}/EV_{}_exec_{}.csv".format(problem, 1, 1), sep=',', header=True, decimal='.', index=False, float_format='%.5f')
# mnist = fetch_openml('mnist_784')
mnist = fetch_openml(name="Fashion-MNIST")

train = [ [torch.from_numpy( np.array(mnist.data[i]) ), torch.tensor(int(mnist.target[i])).long() ] for i in range(1500) ]
valid = [ [torch.from_numpy( np.array(mnist.data[i]) ), torch.tensor(int(mnist.target[i])).long() ] for i in range(1500,1700) ]
test = [ [torch.from_numpy( np.array(mnist.data[i]) ),  torch.tensor(int(mnist.target[i])).long() ] for i in range(1700,1900) ]

# train = torch.utils.data.Subset(train_loader, np.random.choice(7000, 1500, replace=False))
# valid = torch.utils.data.Subset(validation_loader, np.random.choice(200, 200, replace=False))
# test = torch.utils.data.Subset(test_loader, np.random.choice(200, 200, replace=False))

# === for data reading ===

batch_size = 20#64
# === for the genetic algorithm ===

toolbox = base.Toolbox
pop_numb = 20
mutation_prob = 0.5
mutation_happen_prob = 0.5
mate_happen_prob = 0.3
tournsize = 3
learning_rate = 0.1
generation_numb = 30

fc_layer_numb_min_lim = 1
fc_layer_numb_max_lim = 2


dropout_min_lim = 0.5
dropout_max_lim = 0.7


neuron_min_lim = 1
neuron_max_lim = 50

neuron_min_lim = 1
neuron_max_lim = 50

lr_max_lim = 0.001
lr_min_lim = 0.00001

epochs_min_lim = 1
epochs_max_lim = 10



hidden_fc_layers = [5,5]
epochs = 20
dropout = [0.5,0.5]
print_every = 5
patience = 5
act_functions_ref = "relu_all"
optim_ref = "adam"
criter_ref = "nlll"



file = sys.argv[1]
json_file = sys.argv[2]

patience_min_lim = 1
patience_max_lim = 10


conv_layers = [1, 16, 32]
kernel_sizes = [[3,2], [3,2]]
stride_sizes = [1, 1]

kernels_min_lim = 10
kernels_max_lim = 150

possible_kernels_conv = [3, 5, 7]
possible_kernels_pool = [2]
possible_strides_conv = [1,2]
possible_conv_layers_sizes = [3,4,5]

img_size = 28
mutate_all_hparams = False#True
mate_all_hparams = False#True


# "hddn_fc_lyrs","lr","epcs","pat","hddn_fc_lyrs_sz" #cnvl_lyrs, cnv_lrys_sz, krnl_szs, cnv_strd_szs
evol_hparameters = ["hddn_fc_lyrs", "lr", "act_fncs", "cnvl_lyrs", "krnl_szs", "cnv_strd_szs"] 


# transform = transforms.Compose([#torchvision.transforms.Resize(img_size), transforms.ToTensor(),
# 								transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))#transforms.Normalize((0.5,), (0.5,)),
# 								])


# === BEGIN MNIST ===
transform = transforms.Compose([#torchvision.transforms.Resize(50),
								transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5,)) ])

train_sampler = SubsetRandomSampler([i for i in range(1500)])
valid_sampler = SubsetRandomSampler([i for i in range(1500,1700)])
test_sampler = SubsetRandomSampler([i for i in range(1700,1900)])
# === BEGIN MNIST ===



# torch.utils.data.random_split()
# mnist = datasets.MNIST('', train=True, download=True, transform=transform)
#fetch_openml('mnist_784')
# === END MNIST ===


# === BEGIN CIFAR10 ===
# train = datasets.CIFAR10('', train=True, download=True, transform=transform)

# 		# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# valid = datasets.CIFAR10('', train=False, download=True, transform=transform)

# test = datasets.CIFAR10('', train=False, download=True, transform=transform)

# === END CIFAR10 ===

# === BEGIN FashionMNIST ===
# fashion = datasets.FashionMNIST('', train=True, download=True, transform=transform)
# 						transform=transform)
# === END FashionMNIST ===
# === BEGIN Cut ===
trainset = torch.utils.data.DataLoader(train, batch_size=batch_size, drop_last=True)#, sampler=train_sampler)
validset = torch.utils.data.DataLoader(valid, batch_size=batch_size, drop_last=True)#, sampler=valid_sampler)
testset = torch.utils.data.DataLoader(test, batch_size=batch_size, drop_last=True)#, sampler=test_sampler)

# === END Cut ===
# This is the shape of the output that Pytorch expects [B, C, H, W]

wanted_fitness = None#0.75
input_dim = 28
output_dim = 10#len(dataset.NN_LABELS)
# output_dim = len(dataset.NN_LABELS)
genetic_algorithm = GenAlg_CNN(GenAlg_CNN_Descriptor(toolbox, neuron_min_lim, neuron_max_lim, lr_min_lim, \
		lr_max_lim, epochs_min_lim, epochs_max_lim, patience_min_lim, patience_max_lim,\
		fc_layer_numb_min_lim, fc_layer_numb_max_lim, dropout_min_lim, dropout_max_lim, pop_numb, mutation_prob,\
		mutation_happen_prob, mutate_all_hparams, mate_happen_prob, mate_all_hparams, tournsize, generation_numb,\
		evol_hparameters, wanted_fitness, hidden_fc_layers, input_dim, output_dim, act_functions_ref, dropout, \
		batch_size, epochs, learning_rate, optim_ref, criter_ref, print_every, patience, kernels_min_lim, kernels_max_lim, \
		possible_kernels_conv, possible_kernels_pool, possible_strides_conv, possible_conv_layers_sizes,conv_layers, kernel_sizes, stride_sizes))

start_time = time.time()
# genetic_algorithm.simple_genetic_algorithm(dataset.training_data, dataset.validation_data)
genetic_algorithm.simple_genetic_algorithm(trainset, validset)
print("Execution time: ", time.time()-start_time, " seconds | ", (time.time()-start_time)/60, " minutes" )