import warnings,os,sys
warnings.simplefilter('ignore')
import random,time,copy
from deap import base
from deap import creator
from deap import tools
from itertools import repeat
from Network import CNN_Network
from Network import CNN_Descriptor
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets
import copy
import cv2
import numpy as np
from tqdm import tqdm
import json as json
from sklearn.preprocessing import MinMaxScaler

# === BEGIN DELETE AFTERWARDS ===
class Dataset_CNN():

	TRAINING = ""
	VALIDATION = ""
	TESTING = ""
	LABELS = {} # Labels as they are put in the folder
	NN_LABELS = {} # Labels if they begin from 1,2,3 or another number. The secuence must be of order 1
	img_size = None
	batch_size = 1
	training_data = []
	validation_data = []
	testing_data = []

	
	def __init__(self, dir_name, img_size, batch_size):
		self.TRAINING = dir_name + "/train/"
		self.VALIDATION = dir_name + "/valid/"
		self.TESTING = dir_name + "/test/"
		self.img_size = img_size
		self.batch_size = batch_size

	def get_NN_labels(self,json_file_str): #THIS FUNCTION HAS TO BE USED WHEN DEALING WITH THE CLASSIFICATION, NOT WHEN OBTAINING THE DATA
		if len(self.LABELS) == 0:
			with open(json_file_str) as json_file:
				self.LABELS = json.load(json_file)

		has_class_cero = False

		for label in self.LABELS:
			if int(label) == 0:
				self.NN_LABELS = copy.copy(self.LABELS)
				has_class_cero = True
				break

		_min_ = float('Inf')
		for label in self.LABELS:
			if int(label) < _min_:
				_min_ = int(label)

		if not has_class_cero:
			for label in self.LABELS:
				self.NN_LABELS[str((int(label)-_min_))] = self.LABELS[label]
			return _min_
		else: return -1

	def make_data(self, json_file_str, dtloader_str):
		with open(json_file_str) as json_file:
			self.LABELS = json.load(json_file)

		_min_ = self.get_NN_labels(json_file)

		lim_1 = None
		lim_2 = None
		lim_3 = None

		# lim_1 = 1000#5
		# lim_2 = 250#2
		# lim_3 = 250#1
		
		count = 0
		count1 = 0
		count2 = 0
		count3 = 0
		for label in self.LABELS:
			for f in os.listdir(self.TRAINING+label):
				if count == lim_1 and lim_1 != None:
					break
				if "jpg" in f:
					try:
						img_path = "{}/{}/{}".format(self.TRAINING,label,f)
						img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
						img = cv2.resize(img, (self.img_size, self.img_size))

						if _min_ == -1:
							# self.training_data.append([np.array(img),int(label)])
							_label_ = copy.copy(int(label))
							self.training_data.append([img, np.array(_label_)])#torch.Tensor(label)])
						else:
							# self.training_data.append([np.array(img),(int(label)-_min_)])
							_label_ = copy.copy( int(label)-_min_ )
							self.training_data.append([img, np.array(_label_)])#torch.Tensor(label)-_min_])
					except Exception as e:
						print("The following error occurred:\n"+str(e))
						pass
				count+=1

			count = 0
			for f in os.listdir(self.VALIDATION+label):
				if count == lim_2 and lim_2 != None:
					break 
				if "jpg" in f:
					try:
						img_path = "{}/{}/{}".format(self.VALIDATION,label,f)
						img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
						img = cv2.resize(img, (self.img_size, self.img_size))

						if _min_ == -1:
							_label_ = copy.copy(int(label))
							# self.validation_data.append([np.array(img),int(label)])
							# self.validation_data.append([torch.from_numpy(np.array(img)).double(),torch.Tensor(label)])
							self.validation_data.append([img, np.array(_label_)])#torch.Tensor(label)])
						else:
							_label_ = copy.copy(int(label)-_min_)
							# self.validation_data.append([np.array(img),(int(label)-_min_)])
							# self.validation_data.append([torch.from_numpy(np.array(img)).double(), torch.Tensor(label)-_min_])
							self.validation_data.append([img, np.array(_label_)])#torch.Tensor(label)])					
					except Exception as e:
						print("The following error occurred:\n"+str(e))
						pass
				count+=1

			count = 0
			for f in os.listdir(self.TESTING+label):
				if count == lim_3 and lim_3 != None:
					break
				if "jpg" in f:
					try:
						img_path = "{}/{}/{}".format(self.TESTING,label,f)
						img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
						img = cv2.resize(img, (self.img_size, self.img_size))

						if _min_ == -1:
							_label_ = copy.copy(int(label))
							# self.testing_data.append([np.array(img),int(label)])
							# self.testing_data.append([torch.from_numpy(np.array(img)).double(), torch.Tensor(label)])
							self.testing_data.append([img, np.array(_label_)])#torch.Tensor(label)])
						else:
							_label_ = copy.copy(int(label)-_min_)
							# self.testing_data.append([np.array(img),(int(label)-_min_)])
							# self.testing_data.append([torch.from_numpy(np.array(img)).double(), torch.Tensor(label)-_min_])
							self.testing_data.append([img, np.array(_label_)])
					except Exception as e:
						print("The following error occurred:\n"+str(e))
						pass
				count += 1


		values =  [ i[0] for i in self.training_data]

		
		scaler = MinMaxScaler()
		scaler.fit(values)
		values = scaler.transform(values)

		# scaler.fit(self.validation_data)
		# standarized_validloaderr = scaler.transform(self.validation_data)
		train = [ [torch.from_numpy(values[i]).double(), torch.from_numpy(np.array(self.training_data[i][1])).long()] \
			for i in range(len(self.training_data))] #standarized_trainloaderr]

		values =  [ i[0] for i in self.testing_data]

		scaler = MinMaxScaler()
		scaler.fit(values)
		values = scaler.transform(values)

		test = [ [torch.from_numpy(values[i]).double(), torch.from_numpy(np.array(i[1])).long()] \
			for i in range(len(self.testing_data))] #standarized_testloaderr]


		# scaler.fit(self.testing_data)
		# standarized_testloaderr = scaler.transform(self.testing_data)


		# train = [ [torch.from_numpy(np.array(i[0])).double(), torch.from_numpy(np.array(i[1])).long()] \
		# 	for i in self.training_data]#standarized_trainloaderr]

		# valid = [ [torch.from_numpy(np.array(i[0])).double(), torch.from_numpy(np.array(i[1])).long()] \
		# 	for i in self.validation_data]#standarized_validloaderr]

		# test = [ [torch.from_numpy(np.array(i[0])).double(), torch.from_numpy(np.array(i[1])).long()] \
		# 	for i in self.testing_data]#standarized_testloaderr]


		self.training_data = torch.utils.data.DataLoader(train, batch_size = int(self.batch_size), \
			shuffle = True, drop_last=True)
		# self.validation_data = torch.utils.data.DataLoader(valid, batch_size = int(self.batch_size), \
		# 	shuffle = True, drop_last=True)
		self.testing_data = torch.utils.data.DataLoader(test, batch_size = int(self.batch_size),\
			shuffle = True, drop_last=True)

		# print(next(iter(self.training_data)))
		# torch.save(self.training_data, dtloader_str + "_tr")
		# torch.save(self.validation_data, dtloader_str +"_va")
		# torch.save(self.testing_data, dtloader_str +"_te")

	def load_data(self, dtloader_str):
		self.training_data = torch.load(dtloader_str + "_tr")
		self.validation_data = torch.load(dtloader_str +"_va")
		self.testing_data = torch.load(dtloader_str +"_te")

# === for data reading ===


# a = [random.randint(1,10) for i in range(8)]
# b = [random.randint(1,5) for i in range(7)]

# c = [random.randint(1,10) for i in range(11)]
# d = [random.randint(1,5) for i in range(10)]

# index_a = random.randint(0,7)
# while index_a % 2 !=0:
# 	index_a = random.randint(0,7)
# index_b = random.randint(0,6)
# index_c = random.randint(0,10)
# # while index_c % 2 !=0:
# # 	index_c = random.randint(0,10)


# index_d = random.randint(0,9)


# # print("Lista de a de tamaño 8 ", a)
# # print("Lista de b de tamaño 7 ", b)
# # print(index_a, index_b)
# # print(a[0:index_a])
# # print(b[0:(index_a-1)])

# print("Lista de c de tamaño 11 ", c)
# print("Lista de d de tamaño 10 ", d)
# print(c[0:index_c])
# print(d[0:(index_c-1)])
# # print(index_a, index_b, index_c, index_d)
# exit()

n_network = CNN_Network(CNN_Descriptor())
NN_info_file = "Arquitecturas/NN_MNIST_095.txt"
n_network.load_NN_info(NN_info_file)
# === BEGIN MNIST ===
transform = transforms.Compose([#torchvision.transforms.Resize(50),
								transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5,)) ])

# === BEGIN MNIST ===
train = datasets.MNIST('', train=True, download=True,
						transform=transform)

		# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
valid = datasets.MNIST('', train=False, download=True,
						transform=transform)


test = datasets.MNIST('', train=False, download=True,
						transform=transform)
# === END MNIST ===


# === BEGIN CIFAR10 ===
# train = datasets.CIFAR10('', train=True, download=True,
# 						transform=transform)

# 		# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# valid = datasets.CIFAR10('', train=False, download=True,
# 						transform=transform)

# test = datasets.CIFAR10('', train=False, download=True,
# 						transform=transform)

# === END CIFAR10 ===

# === BEGIN FashionMNIST ===
# train = datasets.FashionMNIST('', train=True, download=True,
# 						transform=transform)

# 		# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# valid = datasets.FashionMNIST('', train=True, download=True,
# 						transform=transform)


# test = datasets.FashionMNIST('', train=True, download=True,
# 						transform=transform)
# === END FashionMNIST ===
# print(len(train[0][0][0]))
# exit()

# === BEGIN Cut ===
train = torch.utils.data.Subset(train, np.random.choice(len(train), 10000, replace=False))
valid = torch.utils.data.Subset(valid, np.random.choice(len(valid), 200, replace=False))
test = torch.utils.data.Subset(test, np.random.choice(len(test), 200, replace=False))

trainset = torch.utils.data.DataLoader(train, batch_size=n_network.descriptor.batch_size, shuffle=True)
validset = torch.utils.data.DataLoader(valid, batch_size=n_network.descriptor.batch_size, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=n_network.descriptor.batch_size, shuffle=True)
# === END Cut ===

shuffle = True
# === for the genetic algorithm ===
file = sys.argv[1]
json_file = sys.argv[2]
makenew = True
dtloader_str = "Db_cats_dogs"

# dataset = Dataset_CNN(file, n_network.descriptor.input_dim, n_network.descriptor.input_dim)

# if makenew: dataset.make_data(json_file,dtloader_str)
# else: dataset.load_data(dtloader_str)

start_time = time.time()

n_network.training_CNN(trainset)
with torch.no_grad():
	print("The testing gives an accuracy of ", str( round(100.0 * n_network.testing_CNN(testset), 3 ) ) + " %")
	print("Execution time: ", time.time()-start_time, " seconds | ", (time.time()-start_time)/60, " minutes" )