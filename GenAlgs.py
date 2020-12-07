import warnings,os,sys,random,time,copy
from datetime import datetime
warnings.simplefilter('ignore')
from deap import base
from deap import creator
from deap import tools
from itertools import repeat
from Network import MLP_Network
from Network import MLP_Descriptor
from Network import CNN_Network
from Network import CNN_Descriptor
import matplotlib.pyplot as plt
from statistics import mean, median



class GenAlg_Descriptor:
	# === BEGIN Attributes ===
	# * toolbox = contains the list with the number of neurons for each hidden fully connected layer
	# * neuron_min_lim = minimum number of neurons
	# * neuron_max_lim = maximum number of neurons
	# * lr_min_lim = minimum learning rate
	# * lr_max_lim = maximum learning rate
	# * epochs_min_lim = minimum number of epochs
	# * epochs_max_lim = maximum number of epochs
	# * patience_min_lim = minimum number of patience
	# * patience_max_lim = maximum number of patience
	# * fc_layer_numb = number of fc hidden layers
	# * pop_numb = population size
	# * mutation_prob = probability of a gen being mutated
	# * mutation_happen_prob = probability of an individual being mutated
	# * mate_happen_prob = probability of two individuals being crossed
	# * tournsize = tournament size
	# * generation_numb = number of generations
	# * evol_hparams = hiperparameters that will be evolved
	# * wanted_fitness = fitness that needs to be surpassed or equaled in order to the algorithm to halt
	# === END Attributes ===
	def __init__(self, toolbox, neuron_min_lim, neuron_max_lim, lr_min_lim,lr_max_lim, epochs_min_lim, epochs_max_lim,\
		patience_min_lim, patience_max_lim, fc_layer_numb_min_lim, fc_layer_numb_max_lim, dropout_min_lim, dropout_max_lim, pop_numb,\
		mutation_prob, mutation_happen_prob, mutate_all_hparams, mate_happen_prob, mate_all_hparams, tournsize, generation_numb, evol_hparams, wanted_fitness):

		self.toolbox = toolbox

		self.neuron_min_lim = neuron_min_lim
		self.neuron_max_lim = neuron_max_lim

		self.lr_min_lim = lr_min_lim
		self.lr_max_lim = lr_max_lim

		self.epochs_min_lim = epochs_min_lim
		self.epochs_max_lim = epochs_max_lim

		self.patience_min_lim = patience_min_lim
		self.patience_max_lim = patience_max_lim

		self.fc_layer_numb_min_lim = fc_layer_numb_min_lim
		self.fc_layer_numb_max_lim = fc_layer_numb_max_lim

		self.dropout_min_lim = dropout_min_lim
		self.dropout_max_lim = dropout_max_lim

		self.pop_numb = pop_numb
		self.mutation_prob = mutation_prob
		self.mutation_happen_prob = mutation_happen_prob
		self.mutate_all_hparams = mutate_all_hparams
		
		self.mate_happen_prob = mate_happen_prob
		self.mate_all_hparams = mate_all_hparams
		
		self.tournsize = tournsize
		self.generation_numb = generation_numb

		self.evol_hparams = evol_hparams
		self.wanted_fitness = wanted_fitness

class GenAlg_MLP_Descriptor(GenAlg_Descriptor): #The MLP Genetic algorithm does not have anything that a Genetic Algorithm does not already have
	# === BEGIN Attributes ===
	# * nn_hparameters = hiperparameters of a MLP network
	# === END Attributes ===

	def __init__(self, toolbox = base.Toolbox, neuron_min_lim = 1 ,neuron_max_lim = 50, lr_min_lim = 0.00001, lr_max_lim = 0.1, \
		epochs_min_lim = 1, epochs_max_lim = 200, patience_min_lim=5, patience_max_lim=20, fc_layer_numb_min_lim=1, fc_layer_numb_max_lim=3, dropout_min_lim = 0.0, dropout_max_lim = 1.0,\
		pop_numb=5, mutation_prob=0.5, mutation_happen_prob=0.1, mutate_all_hparams = False, mate_happen_prob=0.5, mate_all_hparams = False, tournsize=3, generation_numb=50, evol_hparams=["hddn_fc_lyrs"], wanted_fitness=None, \
		hidden_fc_layers=[5,5], input_dim = 100, output_dim = 100, act_functions_ref="relu_all", dropout=[0.5,0.5], \
		batch_size=1,  epochs=5, learning_rate=0.01, optim_ref="adam", criter_ref="nlll", print_every=2, patience=5, mode="clssf"):
		# Shall i just put: toolbox = base.Toolbox() ???
		super().__init__(toolbox, neuron_min_lim, neuron_max_lim, lr_min_lim, lr_max_lim, epochs_min_lim, epochs_max_lim,patience_min_lim, patience_max_lim,\
		fc_layer_numb_min_lim, fc_layer_numb_max_lim, dropout_min_lim, dropout_max_lim, pop_numb, mutation_prob, mutation_happen_prob, mutate_all_hparams,\
		mate_happen_prob, mate_all_hparams, tournsize, generation_numb, evol_hparams, wanted_fitness)
		
		self.nn_hparameters = MLP_Descriptor(hidden_fc_layers, input_dim, output_dim, act_functions_ref, dropout, batch_size, \
		epochs, learning_rate, optim_ref, criter_ref, print_every, patience, mode)

class GenAlg_CNN_Descriptor(GenAlg_Descriptor):
	# === BEGIN Attributes ===
	# * kernels_min_lim = The minimum number of kernels on each convolutional layer
	# * kernels_max_lim = The maximum number of kernels on each convolutional layer
	# * kernel_size_min_lim = minimum kernel size
	# * kernel_size_max_lim = maximum kernel size
	# * stride_size_min_lim = minimum stride size
	# * stride_size_max_lim = maximum stride size	
	# * nn_hparameters = hiperparameters of a CNN network
	# === END Attributes ===	
	def __init__(self, toolbox = base.Toolbox, neuron_min_lim = 1, neuron_max_lim = 50, lr_min_lim = 0.00001, \
		lr_max_lim = 0.1, epochs_min_lim = 1, epochs_max_lim = 200, patience_min_lim = 5, patience_max_lim=20,\
		fc_layer_numb_min_lim = 1, fc_layer_numb_max_lim=3, dropout_min_lim = 0.5, dropout_max_lim = 0.5, pop_numb = 5,\
		mutation_prob = .5, mutation_happen_prob = 0.1, mutate_all_hparams = False,mate_happen_prob = 0.5, mate_all_hparams = False,\
		tournsize=3, generation_numb=50, evol_hparams = [0,4], wanted_fitness = None, hidden_fc_layers = [5,5], input_dim = 100, output_dim = 100,\
		act_functions_ref = "relu_all", dropout=[0.5,0.5], batch_size = 1,  epochs = 5, learning_rate = 0.01, optim_ref = "adam", criter_ref = "nlll",\
		print_every=2, patience=5, kernels_min_lim = 32, kernels_max_lim=128, possible_kernels_conv = [3, 5, 7], possible_kernels_pool = [1,2,3],\
		possible_strides_conv = [1,2], possible_conv_layers_sizes=[3,4,5], conv_layers = [1,32,64,128], kernel_sizes=[[3,2],[5,2],[7,2]], conv_stride_sizes=[1,1,1]):

		# super().__init__(toolbox,neuron_min_lim ,neuron_max_lim, lr_min_lim, \
		# lr_max_lim, epochs_min_lim, epochs_max_lim, patience_min_lim, patience_max_lim, \
		# fc_layer_numb_min_lim,fc_layer_numb_max_lim, pop_numb ,mutation_prob, mutation_happen_prob, \
		# mate_happen_prob, tournsize, generation_numb, evol_hparams, wanted_fitness)
		
		super().__init__(toolbox, neuron_min_lim, neuron_max_lim, lr_min_lim, lr_max_lim, epochs_min_lim, epochs_max_lim,patience_min_lim, patience_max_lim,\
		fc_layer_numb_min_lim, fc_layer_numb_max_lim, dropout_min_lim, dropout_max_lim, pop_numb, mutation_prob, mutation_happen_prob, mutate_all_hparams,\
		mate_happen_prob, mate_all_hparams, tournsize, generation_numb, evol_hparams, wanted_fitness)
		
		self.kernels_min_lim = kernels_min_lim 
		self.kernels_max_lim = kernels_max_lim # The maximum number of kernels on each convolutional layer

		self.possible_kernels_conv = possible_kernels_conv
		self.possible_kernels_pool = possible_kernels_pool
		self.possible_strides_conv = possible_strides_conv
		self.possible_conv_layers_sizes = possible_conv_layers_sizes

		self.nn_hparameters = CNN_Descriptor(hidden_fc_layers, input_dim, output_dim, act_functions_ref, dropout, \
		batch_size, epochs, learning_rate, optim_ref, criter_ref, print_every, patience, conv_layers, kernel_sizes, conv_stride_sizes)

# This class represents the fenotipes for MLP networks
class Individual(list):
	# === BEGIN Attributes ===
	# * hidden_fc_layers = hidden fc layers that will be evolved
	# * learning_rate = learning rate that will be evolved
	# * patience = patience that will be evolved
	# * nn = neural network of the individual
	# === END Attributes ===		
	def __init__(self, genAlg):
		# === BEGIN Evolutionable ===
		self.hidden_fc_layers = copy.copy(genAlg.descriptor.nn_hparameters.hidden_fc_layers)
		self.learning_rate = copy.copy(genAlg.descriptor.nn_hparameters.learning_rate)
		self.epochs = copy.copy(genAlg.descriptor.nn_hparameters.epochs)
		self.patience = copy.copy(genAlg.descriptor.nn_hparameters.patience)
		self.hidden_fc_size = copy.copy(len(self.hidden_fc_layers))
		self.act_functions_ref = []
		if genAlg.descriptor.nn_hparameters.act_functions_ref == "relu_all":	
			self.act_functions_ref = ["relu" for i in range(self.hidden_fc_size)]
		elif genAlg.descriptor.nn_hparameters.act_functions_ref == "sigmoid_all":
			self.act_functions_ref = ["sigmoid" for i in range(self.hidden_fc_size)]
		else:
			self.act_functions_ref = copy.copy(genAlg.descriptor.nn_hparameters.act_functions_ref)
		self.dropout = copy.copy(genAlg.descriptor.nn_hparameters.dropout)
		# === END Evolutionable ===
		self.nn = None

		# self.attributes = {0:self.hidden_fc_layers,1:self.learning_rate,2:self.epochs,3:self.patience}	
		# if the chromosomes are among the hiperparameters that will be evolved, they will take random values
		# among the range the user defined
		# To initialize the hiperparameters that will be evolved, first we mutate them
		for i in genAlg.descriptor.evol_hparams:  # ej: [0,2,3]
			if i == "hddn_fc_lyrs":
				self = genAlg.mutate_hddn_fc_lyrs(self)
			if i == "lr":
				self = genAlg.mutate_lr(self)
			if i == "epcs":
				self = genAlg.mutate_epcs(self)
			if i == "pat":
				self = genAlg.mutate_pat(self)
			if i == "act_fncs":
				self = genAlg.mutate_act_fncs(self)
			if i == "hddn_fc_lyrs_sz":
				self = genAlg.mutate_hddn_fc_lyrs_sz(self)
			if i == "drpt":
				self = genAlg.mutate_drpt(self)

	def __str__(self):
		return "hidden_fc_layers: {}\nlearning_rate: {}\nepochs: {}\npatience: {}\nactivation_fncs: {}\nhidden_fc_size: {}\ndropout: {}"\
		.format(self.hidden_fc_layers, self.learning_rate, self.epochs, self.patience, self.act_functions_ref, self.hidden_fc_size, self.dropout)

class GenAlg:
	# With this function we will create a random dropout
	def rand_dropout(self, dropout_min_lim, dropout_max_lim):
		return round(random.uniform(dropout_min_lim, dropout_max_lim),1)

	# 	return init_indiv(self)
	# === BEGIN Attributes ===
	# * descriptor = the descriptor that will be used
	# * toolbox = the toolbox that will be used
	# === END Attributes ===
	# In this function we register the functions that will create random values for the individuals' chromosomes	
	def __init__(self, genAlg_Descriptor):
		self.descriptor = genAlg_Descriptor
		self.toolbox = self.descriptor.toolbox()
		self.toolbox.register("select", tools.selTournament, tournsize=3, fit_attr='fitness')

		self.toolbox.register("select_best", tools.selBest, k=self.descriptor.pop_numb, fit_attr='fitness')

		self.toolbox.register("evaluate", self.func_fitness)
		self.toolbox.register("rand_neuron", random.randint, self.descriptor.neuron_min_lim, self.descriptor.neuron_max_lim)
		self.toolbox.register("rand_act_fnctn", random.randint, 0, 1)
		self.toolbox.register("rand_hddn_fc_lyrs_sz", random.randint, self.descriptor.fc_layer_numb_min_lim, self.descriptor.fc_layer_numb_max_lim)
		self.toolbox.register("rand_drpt",  self.rand_dropout, self.descriptor.dropout_min_lim, self.descriptor.dropout_max_lim)
		
		# It creates a dropout with more than one decimal
		# self.toolbox.register("rand_drpt", random.uniform, self.descriptor.dropout_min_lim, self.descriptor.dropout_max_lim, 1)

		self.toolbox.register("rand_lr", random.uniform, self.descriptor.lr_max_lim, self.descriptor.lr_min_lim)
		self.toolbox.register("rand_epochs", random.randint, self.descriptor.epochs_min_lim , self.descriptor.epochs_max_lim)
		self.toolbox.register("rand_patience", random.randint, self.descriptor.patience_min_lim, self.descriptor.patience_max_lim)

	def func_mate_lists(self,ind1, ind2, boundary_ind):
		
		left_ind1_hidden_fc_layers = ind1.hidden_fc_layers[:boundary_ind] 
		right_ind2_hidden_fc_layers = ind2.hidden_fc_layers[boundary_ind:]
		
		left_ind1_act_functions_ref = ind1.act_functions_ref[:boundary_ind] 
		
		right_ind2_act_functions_ref = ind2.act_functions_ref[boundary_ind:] 

		left_ind1_dropout = ind1.dropout[:boundary_ind] 
		right_ind2_dropout = ind2.dropout[boundary_ind:]

		ind1.hidden_fc_layers[:boundary_ind] = right_ind2_hidden_fc_layers
		ind2.hidden_fc_layers[boundary_ind:] = left_ind1_hidden_fc_layers

		ind1.dropout[:boundary_ind] = right_ind2_dropout
		ind2.dropout[boundary_ind:] = left_ind1_dropout	

		ind1.act_functions_ref[:boundary_ind] = right_ind2_act_functions_ref
		ind2.act_functions_ref[boundary_ind:] = left_ind1_act_functions_ref

		ind1.hidden_fc_size = len(ind1.hidden_fc_layers)
		ind2.hidden_fc_size = len(ind2.hidden_fc_layers)



	def get_current_date_time_str(self):
		now = datetime.now()
		dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
		dt_string = dt_string.replace("/","-")
		dt_string = dt_string.replace(" ","_")
		dt_string = dt_string.replace(":","_")
		return str(dt_string)

	# === BEGIN Write/Read functions ===
	# In this function the neural network is saved
	def save_NN(self, network, dt_string = None, nn_directory = "./"):
		if dt_string == None:
			dt_string = self.get_current_date_time_str()
		network.save_NN(nn_directory+"NN_"+ str(dt_string))
	
	# In this function the neural network's information is saved
	def save_NN_info(self, ind, fitness, dt_string = None, hips_directory = "./"):
		if dt_string == None:
			dt_string = self.get_current_date_time_str()
		ind.nn.save_NN_info(hips_directory+"NN_"+ str(dt_string)+".txt")
		# print("Best fitness: ", fitness)
		# print("Best fitness: ", ind.fitness.values[0])

	# In this function the neural network and its information is saved
	def save_NN_and_info(self, ind, fitness, dt_string = None, hips_nn_directory = "./"):
		if dt_string == None:
			dt_string = self.get_current_date_time_str()		
		ind.nn.save_NN(hips_nn_directory+"NN_"+ str(dt_string))
		ind.nn.save_NN_info(hips_nn_directory+"NN_"+ str(dt_string)+".txt")
		print("Best fitness: ", fitness)
		# print("Best fitness: ", ind.fitness.values[0])

	def show_graphic(self, ax_x, ax_y, title, xlabel, ylabel):
		plt.clf()
		plt.xlim([0.0, ax_x[-1]])
		plt.ylim([0.0, 1.0])

		plt.title(title)
		# plt.scatter(torch.max(i).item(),labels[idx].item(),label="Result",alpha=0.5)
		plt.scatter(ax_x, ax_y)
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.legend()
		# plt.savefig("aaaaa")		
		plt.show()


	def save_graphic(self, ax_x, ax_y, title, xlabel, ylabel, img_name = None, pic_directory = "./"):
		plt.clf()
		plt.xlim([0.0, ax_x[-1]])
		plt.ylim([0.0, 1.0])
		if img_name == None:
			img_name = self.get_current_date_time_str()
		plt.title(title)
		# plt.scatter(torch.max(i).item(),labels[idx].item(),label="Result",alpha=0.5)
		plt.scatter(ax_x, ax_y)
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.legend()
		plt.savefig(pic_directory+img_name)



	def get_fitnesses(self, _list_, trainloader, validloader):
		fitnesses = []
		for i in _list_:
			fitnesses.append(self.toolbox.evaluate(i,trainloader,validloader))
		return fitnesses

	def apply_fitnesses(self, applied_pop, fitnesses):
		for ind, fit in zip(applied_pop, fitnesses):
			ind.fitness.values = fit
			ind.nn = fit[1]
		return applied_pop


	def mutate_hddn_fc_lyrs(self, mutate, prob=1.0):
		hidden_fc_layers_no_mutated = copy.copy(mutate.hidden_fc_layers)
		mutate.hidden_fc_layers = []
		for j in range(mutate.hidden_fc_size): # HAY ALGO MAL AQUI
			if random.random() < prob:
				mutate.hidden_fc_layers.append(self.toolbox.rand_neuron())
			else:
				mutate.hidden_fc_layers.append(hidden_fc_layers_no_mutated[j])
		return mutate

	def mutate_lr(self, mutate,prob = 1.0):
		if random.random() < prob:
			mutate.learning_rate = self.toolbox.rand_lr()
		return mutate
	def mutate_epcs(self, mutate, prob = 1.0):
		if random.random() < prob:
			mutate.epochs = self.toolbox.rand_epochs()
		return mutate
	def mutate_pat(self, mutate, prob = 1.0):
		if random.random() < prob:
			mutate.patience = self.toolbox.rand_patience()
		return mutate
	
	def mutate_act_fncs(self, mutate, prob = 1.0):
		act_functions_ref_no_mutated = copy.copy(mutate.act_functions_ref)
		mutate.act_functions_ref = []
		for j in range(mutate.hidden_fc_size): # genAlg.descriptor.fc_layer_numb == len( genAlg.descriptor.nn_hparameters.act_functions_ref)
			if random.random() < prob:
				selection = random.choice(["relu","sigmoid"])#self.toolbox.rand_act_fnctn()
				mutate.act_functions_ref.append(selection)
			else:
				if act_functions_ref_no_mutated[j] == "relu":
					mutate.act_functions_ref.append("relu")
				else:
					mutate.act_functions_ref.append("sigmoid")
		return mutate


	def mutate_hddn_fc_lyrs_sz(self, mutate, prob = 1.0):
		if random.random() < prob:
			mutate.hidden_fc_size = self.toolbox.rand_hddn_fc_lyrs_sz()
			_length_ = len(mutate.hidden_fc_layers)
			
			while mutate.hidden_fc_size < _length_:
				del mutate.hidden_fc_layers[-1]
				del mutate.act_functions_ref[-1]
				del mutate.dropout[-1]
				_length_ = len(mutate.hidden_fc_layers)
		
			while mutate.hidden_fc_size > _length_:
				mutate.hidden_fc_layers.append(self.toolbox.rand_neuron())
				if self.descriptor.nn_hparameters.act_functions_ref == "relu_all":
					mutate.act_functions_ref.append("relu")
				elif self.descriptor.nn_hparameters.act_functions_ref == "sigmoid_all":
					mutate.act_functions_ref.append("sigmoid")
				else:
					selection = random.choice(["relu","sigmoid"])#self.toolbox.rand_act_fnctn()
					mutate.act_functions_ref.append(selection)
				mutate.dropout.append(self.toolbox.rand_drpt())
				_length_ = len(mutate.hidden_fc_layers)
		return mutate


	def mutate_drpt(self, mutate, prob = 1.0):
		dropouts_no_mutated = copy.copy(mutate.dropout)
		mutate.dropout = []
		for j in range(mutate.hidden_fc_size):
			if random.random() < prob:
				mutate.dropout.append(self.toolbox.rand_drpt())
			else:
				mutate.dropout.append(dropouts_no_mutated[j])
		return mutate

	def mate_hddn_fc_lyrs_drpt_act_fncs(self, ind1, ind2, hddn_fc_lyrs_drpt_act_fncs_crossed):
		hddn_fc_lyrs_drpt_act_fncs_crossed = True
		_length_ = min(ind1.hidden_fc_size, ind2.hidden_fc_size)
		if _length_ > 1:
			boundary_ind = random.randint(1,_length_-1)
			self.func_mate_lists(ind1, ind2, boundary_ind)
		else:
			aux_fc = copy.copy(ind1.hidden_fc_layers)
			aux_act_funcs_ref = copy.copy(ind1.act_functions_ref)
			aux_act_dropout = copy.copy(ind1.dropout)

			ind1.hidden_fc_layers = copy.copy(ind2.hidden_fc_layers)
			ind1.act_functions_ref = copy.copy(ind2.act_functions_ref)
			ind1.dropout = copy.copy(ind2.dropout)
			ind1.hidden_fc_size = copy.copy(len(ind1.hidden_fc_layers))

			ind2.hidden_fc_layers = copy.copy(aux_fc)
			ind2.act_functions_ref = copy.copy(aux_act_funcs_ref)
			ind2.dropout = copy.copy(aux_act_dropout)
			ind2.hidden_fc_size = copy.copy(len(ind2.hidden_fc_layers))

		return ind1, ind2, hddn_fc_lyrs_drpt_act_fncs_crossed

	def mate_lr(self, ind1, ind2):
		aux = copy.copy(ind1.learning_rate)
		ind1.learning_rate = copy.copy(ind2.learning_rate)
		ind2.learning_rate = copy.copy(aux)
		return ind1, ind2

	def mate_epcs(self, ind1, ind2):
		aux = copy.copy(ind1.epochs)
		ind1.epochs = copy.copy(ind2.epochs)
		ind2.epochs = copy.copy(aux)
		return ind1, ind2
	def mate_pat(self, ind1, ind2):
		aux = copy.copy(ind1.patience)
		ind1.patience = copy.copy(ind2.patience)
		ind2.patience = copy.copy(aux)
		return ind1, ind2

	def mate_hddn_fc_lyrs_sz(self, ind1, ind2):
		aux = copy.copy(ind1.hidden_fc_size)
		ind1.hidden_fc_size = ind2.hidden_fc_size
		ind2.hidden_fc_size = aux
		
		_length_ = len(ind1.hidden_fc_layers)		
		while ind1.hidden_fc_size < _length_:
			del ind1.hidden_fc_layers[-1]
			del ind1.act_functions_ref[-1]
			del ind1.dropout[-1]
			_length_ = len(ind1.hidden_fc_layers)
		while ind1.hidden_fc_size > _length_:
			ind1.hidden_fc_layers.append(self.toolbox.rand_neuron())
			selection = random.choice(["relu","sigmoid"]) #self.toolbox.rand_act_fnctn()
			ind1.act_functions_ref.append(selection)
			ind1.dropout.append(self.toolbox.rand_drpt())
			_length_ = len(ind1.hidden_fc_layers)

		_length_ = len(ind2.hidden_fc_layers)
		while ind2.hidden_fc_size < _length_:
			del ind2.hidden_fc_layers[-1]
			del ind2.act_functions_ref[-1]
			del ind2.dropout[-1]
			_length_ = len(ind2.hidden_fc_layers)
		while ind2.hidden_fc_size > _length_:
			ind2.hidden_fc_layers.append(self.toolbox.rand_neuron())
			selection = random.choice(["relu","sigmoid"]) #self.toolbox.rand_act_fnctn()
			ind2.act_functions_ref.append(selection)
			ind2.dropout.append(self.toolbox.rand_drpt())
			_length_ = len(ind2.hidden_fc_layers)
		return ind1, ind2


	# In this function we get the individual with the best fitness
	# default mode is 0 (classification) so this function can be used in CNN networks as well
	def get_best_ind_fitness(self, fits, pop, best_fitness, best_ind, mode = "clssf"):
		best_fitnesss = copy.copy(best_fitness) # copy of the best fitness up to this point
		best_indd = copy.copy(best_ind) # copy of the best individual up to this point
		if mode == "clssf":
			if max(fits) > best_fitnesss:
				best_fitnesss = max(fits)
				index_best_ind = fits.index(best_fitnesss)
				best_indd = pop[index_best_ind]
		else:
			if min(fits) < best_fitnesss:
				best_fitnesss = min(fits)
				index_best_ind = fits.index(best_fitnesss)
				best_indd = pop[index_best_ind]
		
		 # if there is no fitness that will make our evolution halt, then it will only stop when the limit
		 # of generations is reached
		if self.descriptor.wanted_fitness == None:
			halt = False
		else:
			# Otherwise it will look if the best fitness is better or equal to the one that the user seeks
			if mode == "clssf": # In mode 0 we want the fitness to be as big as possible
				if best_fitnesss >= self.descriptor.wanted_fitness:
					halt = True
				else:
					halt = False
			else: # In mode 1 we want the fitness to be as small as possible
				if best_fitnesss <= self.descriptor.wanted_fitness:
					halt = True
				else:
					halt = False				
		return best_indd, best_fitnesss, halt

# === END Write/Read function ===

class GenAlg_MLP(GenAlg):
	# The deap functions  related to mutation, crossing, individuals' and populations' instantation will take place 
	def __init__(self, genAlg_Descriptor):
		super().__init__(genAlg_Descriptor)
		if self.descriptor.nn_hparameters.mode == "clssf":
			creator.create("Fitness", base.Fitness, weights=(1.0,))
		else:
			creator.create("Fitness", base.Fitness, weights=(-1.0,))
		creator.create("Individual", Individual, fitness=creator.Fitness) # New one
		# We define the individual for a MLP genetic algorithm
		# Alias of the function + constructor of the class + genetic algorithm
		self.toolbox.register("individual", creator.Individual, self)
		# We define the population for a MLP genetic algorithm
		
		self.toolbox.register("population_MLP", tools.initRepeat, list, \
			self.toolbox.individual, self.descriptor.pop_numb)
		
		# May i change them to mate and mutate and put them in Gen_Alg?
		self.toolbox.register("mate_MLP", self.func_mate_MLP, mate_all_hparams = self.descriptor.mate_all_hparams)
		self.toolbox.register("mutate_MLP", self.func_mutation_MLP, prob = self.descriptor.mutation_prob, \
			mutate_all_hparams = self.descriptor.mutate_all_hparams)

		self.toolbox.register("hallOfFame", tools.HallOfFame, maxsize = self.descriptor.pop_numb)
	# In this function a MLP Individual will be mutated
	# if mutate all hparams is true all the chromosomes will suffer mutation,
	# otherwise just a random chromosome
	

	def func_mutation_MLP(self, prob, mutate, mutate_all_hparams=False):
		# print("Mutation took place!")
		if mutate_all_hparams:
			for i in self.descriptor.evol_hparams:
				if i == "hddn_fc_lyrs":
					mutate = self.mutate_hddn_fc_lyrs(mutate, prob)
				if i == "lr":
					mutate = self.mutate_lr(mutate, prob)
				if i == "epcs":
					mutate = self.mutate_epcs(mutate, prob)
				if i == "pat":
					mutate = self.mutate_pat(mutate, prob)
				if i == "act_fncs":
					mutate = self.mutate_act_fncs(mutate, prob)
				if i == "hddn_fc_lyrs_sz":
					mutate = self.mutate_hddn_fc_lyrs_sz(mutate, prob)
				if i == "drpt":
					mutate = self.mutate_drpt(mutate, prob)
		else:
			characteristic = random.choice(self.descriptor.evol_hparams)
			if characteristic == "hddn_fc_lyrs":
				mutate = self.mutate_hddn_fc_lyrs(mutate, prob)
			if characteristic == "lr":
				mutate = self.mutate_lr(mutate, prob)
			if characteristic == "epcs":
				mutate = self.mutate_epcs(mutate, prob)
			if characteristic == "pat":
				mutate = self.mutate_pat(mutate, prob)
			if characteristic == "act_fncs":
				mutate = self.mutate_act_fncs(mutate, prob)
			if characteristic == "hddn_fc_lyrs_sz":
				mutate = self.mutate_hddn_fc_lyrs_sz(mutate, prob)
			if characteristic == "drpt":
				mutate = self.mutate_drpt(mutate, prob)	

	def func_mate_MLP(self, ind1, ind2, mate_all_hparams=True):
		# print("Crossing took place")
		hddn_fc_lyrs_drpt_act_fncs_crossed = False
		if mate_all_hparams:
			for i in self.descriptor.evol_hparams:  # ej: [0,2,3]
				if (i == "hddn_fc_lyrs" or i == "drpt" or i == "act_fncs") and not hddn_fc_lyrs_drpt_act_fncs_crossed:
					ind1, ind2, hddn_fc_lyrs_drpt_act_fncs_crossed = self.mate_hddn_fc_lyrs_drpt_act_fncs(ind1,ind2,hddn_fc_lyrs_drpt_act_fncs_crossed)					
				if i == "lr":
					ind1, ind2 = self.mate_lr(ind1, ind2)
				if i == "epcs":
					ind1, ind2 = self.mate_epcs(ind1, ind2)
				if i == "pat":
					ind1, ind2 = self.mate_pat(ind1, ind2)
				if i == "hddn_fc_lyrs_sz":
					ind1, ind2 = self.mate_hddn_fc_lyrs_sz(ind1, ind2)
		else:
			characteristic = random.choice(self.descriptor.evol_hparams)
			if (characteristic == "hddn_fc_lyrs" or characteristic == "drpt" or characteristic == "act_fncs")\
			 and not hddn_fc_lyrs_drpt_act_fncs_crossed:
				ind1, ind2, hddn_fc_lyrs_drpt_act_fncs_crossed = self.mate_hddn_fc_lyrs_drpt_act_fncs(ind1,ind2,hddn_fc_lyrs_drpt_act_fncs_crossed)					
			if characteristic == "lr":
				ind1, ind2 = self.mate_lr(ind1, ind2)
			if characteristic == "epcs":
				ind1, ind2 = self.mate_epcs(ind1, ind2)
			if characteristic == "pat":
				ind1, ind2 = self.mate_pat(ind1, ind2)
			if characteristic == "hddn_fc_lyrs_sz":
				ind1, ind2 = self.mate_hddn_fc_lyrs_sz(ind1, ind2)
	
	# In this function the MLP network is initialized and returned
	def initialize_NN(self, descriptor_MLP):
		network = MLP_Network(descriptor_MLP)
		return network

	# In this function the MLP network is trained and validated
	def execute_NN(self,network,trainloader,validloader):
		network.training_MLP(trainloader)
		return network.testing_MLP(validloader)

	# In this function the MLP network's fitness is obtained
	def func_fitness(self,individual,trainloader,validloader):
		hidden_fc_layers = individual.hidden_fc_layers #self.descriptor.nn_hparameters.hidden_fc_layers
		num_of_inputs = self.descriptor.nn_hparameters.input_dim
		num_of_outputs = self.descriptor.nn_hparameters.output_dim
		act_functions_ref = individual.act_functions_ref#self.descriptor.nn_hparameters.act_functions_ref
		dropout = individual.dropout#self.descriptor.nn_hparameters.dropout
		batch_size = self.descriptor.nn_hparameters.batch_size
		epochs = individual.epochs#self.descriptor.nn_hparameters.epochs
		learning_rate = individual.learning_rate#self.descriptor.nn_hparameters.learning_rate
		optim_ref = self.descriptor.nn_hparameters.optim_ref
		criter_ref = self.descriptor.nn_hparameters.criter_ref
		patience = individual.patience#self.descriptor.nn_hparameters.patience
		
		try:
			network = self.initialize_NN(MLP_Descriptor(hidden_fc_layers, num_of_inputs, num_of_outputs, \
				act_functions_ref, dropout, batch_size, epochs, learning_rate,optim_ref, \
				criter_ref, self.descriptor.nn_hparameters.print_every, \
				patience, self.descriptor.nn_hparameters.mode ))

			ftnss = self.execute_NN(network,trainloader, validloader)
			return ftnss, network
		# If an error occurs, it will return -Infinity along with the default CNN
		except Exception as e:
			# print(e)
			return float("-Inf"), MLP_Network(MLP_Descriptor())

	def mate_offspring_MLP(self, offspring):
		# We mate the individuals
		for child1, child2 in zip(offspring[::2], offspring[1::2]):
			# print("\n\n==============")
			# print("Children antes \n", child1, "\n", child2)
			# print("-------------")
			if random.random() < self.descriptor.mate_happen_prob:
				self.toolbox.mate_MLP(child1,child2)
				# print("Children despues \n", child1, "\n",child2)
				# print("==============\n\n")
				del child1.fitness.values
				del child2.fitness.values
		return offspring

	def mutate_offspring_MLP(self, offspring):
		for mutant in offspring:
			# print("\n\n==============")
			# print("Individual antes\n", mutant)
			# print("-------------")
			if random.random() < self.descriptor.mutation_happen_prob:
				self.toolbox.mutate_MLP(mutate=mutant)
				# print("Individual despues\n", mutant)
				# print("==============\n\n")
				del mutant.fitness.values
		return offspring




	# In this function we execute the genetic algorithm
	def simple_genetic_algorithm(self,trainloader,validloader, pic_name = None, pic_directory="./"):
		# This variable will define the best fitness (best result) among all the fenotipes in all the generations
		#best_fitness = None
		best_ind = None
		if self.descriptor.nn_hparameters.mode == "clssf":
			best_fitness = float('-Inf')
		else:
			best_fitness = float('Inf')
		
		# BEGIN PROVISIONAL
		best_fitnesses = []
		mean_fitnesses = []
		times = []
		# END PROVISIONAL
		# We create the population
		pop = self.toolbox.population_MLP()

		fitnesses = []
		fitnesses = self.get_fitnesses(pop, trainloader, validloader)

		pop = self.apply_fitnesses(pop, fitnesses)

		# We get the accuracies of all the individuals within the population
		fits = [ind.fitness.values[0] for ind in pop]
		
		g = 0
		# We get the individual that has proven to give the highest accuracy
		best_ind, best_fitness, halt = self.get_best_ind_fitness(fits, pop, best_fitness, best_ind, \
			self.descriptor.nn_hparameters.mode)	

		while g < self.descriptor.generation_numb and not halt:
			start_time = time.time()
			#Create a new generation
			g += 1
			# print(" -- Generation %i --" %g)
			# We select the individuals for the new generation
			# offspring = self.toolbox.select(pop,len(pop))
			offspring = self.toolbox.select(pop,len(pop))

			# We clone the individuals for the new generation
			offspring = list(map(self.toolbox.clone, offspring))

			# We mate the individuals
			offspring = self.mate_offspring_MLP(offspring)
			# We mutate the individuals
			offspring = self.mutate_offspring_MLP(offspring)

			# We evaluate the individuals with an invalid fitness, which means they have been modified
			# If we have deleted its fitness the value of ind.fitness stops being valid
			
			invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

			fitnesses = []
			# We get the new fitnesses for the modified individuals
			fitnesses = self.get_fitnesses(invalid_ind, trainloader, validloader)
			# We get the new fitnesses for the modified individuals
			invalid_ind = self.apply_fitnesses(invalid_ind, fitnesses)

			pop_and_offspring = pop + offspring

			# pop[:] = offspring
			pop[:] = self.toolbox.select_best(pop_and_offspring)

			fits = [ind.fitness.values[0] for ind in pop]

			if self.descriptor.nn_hparameters.mode == "clssf":
				best_fitnesses.append(max(fits))
			else:
				best_fitnesses.append(min(fits))
						
			mean_fitnesses.append(mean(fits))
			times.append(time.time()-start_time)
			if len(list(fits)) > 0:
				best_ind, best_fitness, halt = self.get_best_ind_fitness(fits, pop, best_fitness, best_ind,\
					self.descriptor.nn_hparameters.mode)
			# We save the NN and its info
		# self.show_graphic([i for i in range(g)], best_fitnesses, "Evolution of the best fitness","Generation", "Best fitness")
		
		if pic_name == None:
			nn_and_pic_name = self.get_current_date_time_str()
		else:
			nn_and_pic_name = pic_name
		# self.save_graphic([i for i in range(g)], best_fitnesses, "Evolution of the best fitness","Generation", "Best fitness", nn_and_pic_name , pic_directory)
		self.save_NN_and_info(best_ind, best_fitness, nn_and_pic_name, pic_directory)
		return [i for i in range(g)], best_fitnesses, mean_fitnesses, times




# This class represents the fenotipes for CNN networks
class Individual_CNN(Individual):
	# === BEGIN Attributes ===
	# * conv_layers = convolutional layers that will be evolved
	# * img_size = image size that will be evolved
	# * kernel_size = kernel size that will be evolved
	# * stride_size = stride size that will be evolved
	# === END Attributes ===	
	
	# In this function we register the functions that will create random values for the individuals' chromosomes	
	def __init__(self, genAlg):
		super().__init__(genAlg)
		self.conv_layers = genAlg.descriptor.nn_hparameters.conv_layers
		self.conv_layers_size = len(genAlg.descriptor.nn_hparameters.conv_layers)
		self.kernel_sizes = genAlg.descriptor.nn_hparameters.kernel_sizes
		self.conv_stride_sizes = genAlg.descriptor.nn_hparameters.conv_stride_sizes
		for i in genAlg.descriptor.evol_hparams:  # ej: [0,2,3]
			if i == "cnv_lyrs":
				self = genAlg.mutate_cnv_lyrs(self)
			if i == "cnv_lyrs_sz":
				self = genAlg.mutate_cnv_lyrs_sz(self)
			if i == "krnl_szs":
				self = genAlg.mutate_krnl_szs(self)
			if i == "cnv_strd_szs":
				self = genAlg.mutate_cnv_strd_szs(self)

	def __str__(self):
		return  "{}\nconv_layers: {}\nkernel_sizes: {}\nstride_sizes: {}".format(\
			super().__str__(), self.conv_layers, self.kernel_sizes, self.conv_stride_sizes)

class GenAlg_CNN(GenAlg):

	def __init__(self, genAlg_Descriptor):
		super().__init__(genAlg_Descriptor)
		creator.create("Fitness", base.Fitness, weights=(1.0,))
		creator.create("Individual_CNN", Individual_CNN, fitness=creator.Fitness) # New one

		self.toolbox.register("individual_CNN", creator.Individual_CNN, self)

		self.toolbox.register("population_CNN", tools.initRepeat, list, \
			self.toolbox.individual_CNN, self.descriptor.pop_numb)

		self.toolbox.register("evaluate", self.func_fitness)
		self.toolbox.register("mate_CNN",self.func_mate_CNN, mate_all_hparams = self.descriptor.mate_all_hparams)
		self.toolbox.register("mutate_CNN", self.func_mutation_CNN, \
			prob = self.descriptor.mutation_prob, mutate_all_hparams = self.descriptor.mutate_all_hparams)

		self.toolbox.register("rand_kernel_numb",  random.randint, self.descriptor.kernels_min_lim,\
			self.descriptor.kernels_max_lim)

		self.toolbox.register("rand_conv_layers_size",  self.rand_conv_layers_size)
		
		self.toolbox.register("rand_kernel_sizes", self.rand_kernel_sizes)

		self.toolbox.register("rand_conv_stride_sizes", self.rand_conv_stride_sizes)

	def mutate_cnv_lyrs(self, mutate, prob = 1.0):
		conv_layers_no_mutated = copy.copy(mutate.conv_layers)
		mutate.conv_layers = [1]
		for j in range(1,mutate.conv_layers_size):
			if random.random() < prob:
				kernel_numb = self.toolbox.rand_kernel_numb()
				while kernel_numb % 8 != 0:
					kernel_numb += 1
				mutate.conv_layers.append(kernel_numb)
			else:
				mutate.conv_layers.append(conv_layers_no_mutated[j])
		return mutate

	def rand_kernel_sizes(self):
		return [random.choice(self.descriptor.possible_kernels_conv), \
		random.choice(self.descriptor.possible_kernels_pool)]
	
	def rand_conv_stride_sizes(self):
		return random.choice(self.descriptor.possible_strides_conv)

	def rand_conv_layers_size(self):
		return random.choice(self.descriptor.possible_conv_layers_sizes)

	def mutate_cnv_lyrs_sz(self, mutate, prob = 1.0):
		if random.random() < prob:
			mutate.conv_layers_size = self.toolbox.rand_conv_layers_size()
			_length_ = len(mutate.conv_layers)
			
			while mutate.conv_layers_size < _length_:
				del mutate.conv_layers[-1]
				_length_ = len(mutate.conv_layers)
		
			while mutate.conv_layers_size > _length_:
				mutate.conv_layers.append(self.toolbox.rand_kernel_numb())
				_length_ = len(mutate.conv_layers)

			# Now we update the kernel sizes' and stride sizes' lists, which must have a length of 
			# len(conv_layers)-1
			_length_ = len(mutate.kernel_sizes)
			
			while mutate.conv_layers_size-1 < _length_:
				del mutate.kernel_sizes[-1]
				del mutate.conv_stride_sizes[-1]
				_length_ = len(mutate.kernel_sizes)
			
			while mutate.conv_layers_size-1 > _length_:
				mutate.kernel_sizes.append(self.rand_kernel_sizes())
				mutate.conv_stride_sizes.append(self.rand_conv_stride_sizes())
				_length_ = len(mutate.kernel_sizes)
		

		return mutate

	def mutate_krnl_szs(self, mutate, prob = 1.0):
		kernel_sizes_no_mutated = copy.copy(mutate.kernel_sizes)
		mutate.kernel_sizes = []
		for i in range(mutate.conv_layers_size-1):
			if random.random() < prob:
				mutate.kernel_sizes.append(self.rand_kernel_sizes())
			else:
				mutate.kernel_sizes.append(kernel_sizes_no_mutated[i])
		return mutate

	def mutate_cnv_strd_szs(self, mutate, prob = 1.0):
		conv_stride_sizes_no_mutated = copy.copy(mutate.conv_stride_sizes)
		mutate.conv_stride_sizes = []
		for i in range(mutate.conv_layers_size-1):
			if random.random() < prob:
				mutate.conv_stride_sizes.append(self.rand_conv_stride_sizes())
			else:
				mutate.conv_stride_sizes.append(conv_stride_sizes_no_mutated[i])
		return mutate
	
	def func_mate_conv(self, ind1, ind2, boundary_ind): #En la mutacion se pasan de arriba a abajo y viceversa (los tamainos no cambian)
		_length_ = min(ind1.conv_layers_size, ind2.conv_layers_size)
		right_ind_conv1 = ind1.conv_layers[boundary_ind:]
		right_ind_conv2 = ind2.conv_layers[boundary_ind:]

		ind1.conv_layers[boundary_ind:] = right_ind_conv2
		ind2.conv_layers[boundary_ind:] = right_ind_conv1

		boundary_ind_kernels_strides = boundary_ind-1

		right_ind_kernel1 = ind1.kernel_sizes[boundary_ind_kernels_strides:]
		right_ind_kernel2 = ind2.kernel_sizes[boundary_ind_kernels_strides:]

		right_ind_strides1 = ind1.conv_stride_sizes[boundary_ind_kernels_strides:]
		right_ind_strides2 = ind2.conv_stride_sizes[boundary_ind_kernels_strides:]

		ind1.kernel_sizes[boundary_ind_kernels_strides:] = right_ind_kernel2
		ind2.kernel_sizes[boundary_ind_kernels_strides:] = right_ind_kernel1
		ind1.conv_stride_sizes[boundary_ind_kernels_strides:] = right_ind_strides2
		ind2.conv_stride_sizes[boundary_ind_kernels_strides:] = right_ind_strides1



	def mate_cnv_lyrs_krnl_szs_cnv_strd_szs(self, ind1, ind2, cnv_lyrs_krnl_szs_cnv_strd_szs_crossed):
		cnv_lyrs_krnl_szs_cnv_strd_szs_crossed = True
		_length_ = min(ind1.conv_layers_size, ind2.conv_layers_size)
		if _length_ > 1:
			boundary_ind = random.randint(1,_length_-1)
			self.func_mate_conv(ind1, ind2, boundary_ind)
		else:
			aux_conv = copy.copy(ind1.conv_layers)
			aux_kernels = copy.copy(ind1.kernel_sizes)
			aux_strides = copy.copy(ind1.conv_stride_sizes)

			ind1.conv_layers = copy.copy(ind2.conv_layers)
			ind1.kernel_sizes = copy.copy(ind2.kernel_sizes)
			ind1.conv_stride_sizes = copy.copy(ind2.conv_stride_sizes)
			ind1.conv_layers_size = copy.copy(len(ind1.conv_layers))

			ind2.conv_layers = copy.copy(aux_conv)
			ind2.kernel_sizes = copy.copy(aux_kernels)
			ind2.conv_stride_sizes = copy.copy(aux_strides)
			ind2.conv_layers_size = copy.copy(len(ind2.conv_layers))
	
		return ind1, ind2, cnv_lyrs_krnl_szs_cnv_strd_szs_crossed

	def mate_cnv_lyrs_sz(self, ind1, ind2):
		aux = copy.copy(ind1.conv_layers_size)
		ind1.conv_layers_size = ind2.conv_layers_size
		ind2.conv_layers_size = aux

		_length_ = len(ind1.conv_layers)
		while ind1.conv_layers_size < _length_:
			del ind1.conv_layers[-1]
			_length_ = len(ind1.conv_layers)
		while ind1.conv_layers_size > _length_:
			ind1.conv_layers.append(self.toolbox.rand_kernel_numb())
			_length_ = len(ind1.conv_layers)

		_length_ = len(ind2.conv_layers)
		while ind2.conv_layers_size < _length_:
			del ind2.conv_layers[-1]
			_length_ = len(ind2.conv_layers)			
		while ind2.conv_layers_size > _length_:
			ind2.conv_layers.append(self.toolbox.rand_kernel_numb())
			_length_ = len(ind2.conv_layers)

		# Now we update the kernel sizes' and stride sizes' lists, which must have a length of 
		# len(conv_layers)-1
		_length_ = len(ind1.kernel_sizes)
		while ind1.conv_layers_size-1 < _length_:
			del ind1.kernel_sizes[-1]
			del ind1.conv_stride_sizes[-1]
			_length_ = len(ind1.kernel_sizes)
		
		while ind1.conv_layers_size-1 > _length_:
			ind1.kernel_sizes.append(self.rand_kernel_sizes())
			ind1.conv_stride_sizes.append(self.toolbox.rand_conv_stride_sizes())
			_length_ = len(ind1.kernel_sizes)
		
		_length_ = len(ind2.kernel_sizes)
		while ind2.conv_layers_size-1 < _length_:
			del ind2.kernel_sizes[-1]
			del ind2.conv_stride_sizes[-1]
			_length_ = len(ind2.kernel_sizes)
		
		while ind2.conv_layers_size-1 > _length_:
			ind2.kernel_sizes.append(self.rand_kernel_sizes())
			ind2.conv_stride_sizes.append(self.toolbox.rand_conv_stride_sizes())
			_length_ = len(ind2.kernel_sizes)
	
		return ind1, ind2

	# In this function a CNN Individual will be mutated
	# if mutate all hparams is true all the chromosomes will suffer mutation,
	# otherwise just a random chromosome
	def func_mutation_CNN(self, prob, mutate, mutate_all_hparams=False):
		# print("Mutation took place!")
		if mutate_all_hparams:
			for i in self.descriptor.evol_hparams:
				if i == "hddn_fc_lyrs":
					mutate = self.mutate_hddn_fc_lyrs(mutate, prob)
				if i == "lr":
					mutate = self.mutate_lr(mutate, prob)
				if i == "epcs":
					mutate = self.mutate_epcs(mutate, prob)
				if i == "pat":
					mutate = self.mutate_pat(mutate, prob)
				if i == "act_fncs":
					mutate = self.mutate_act_fncs(mutate, prob)
				if i == "hddn_fc_lyrs_sz":
					mutate = self.mutate_hddn_fc_lyrs_sz(mutate, prob)
				if i == "drpt":
					mutate = self.mutate_drpt(mutate, prob)
				if i == "cnv_lyrs":
					mutate = self.mutate_cnv_lyrs(mutate, prob)
				if i == "cnv_lyrs_sz":
					mutate = self.mutate_cnv_lyrs_sz(mutate, prob)
				if i == "krnl_szs":
					mutate = self.mutate_krnl_szs(mutate, prob)
				if i == "cnv_strd_szs":
					mutate = self.mutate_cnv_strd_szs(mutate, prob)
		else:
			characteristic = random.choice(self.descriptor.evol_hparams)
			if characteristic == "hddn_fc_lyrs":
				mutate = self.mutate_hddn_fc_lyrs(mutate, prob)
			if characteristic == "lr":
				mutate = self.mutate_lr(mutate, prob)
			if characteristic == "epcs":
				mutate = self.mutate_epcs(mutate, prob)
			if characteristic == "pat":
				mutate = self.mutate_pat(mutate, prob)
			if characteristic == "act_fncs":
				mutate = self.mutate_act_fncs(mutate, prob)
			if characteristic == "hddn_fc_lyrs_sz":
				mutate = self.mutate_hddn_fc_lyrs_sz(mutate, prob)
			if characteristic == "drpt":
				mutate = self.mutate_drpt(mutate, prob)
			if characteristic == "cnv_lyrs":
				mutate = self.mutate_cnv_lyrs(mutate, prob)
			if characteristic == "cnv_lyrs_sz":
				mutate = self.mutate_cnv_lyrs_sz(mutate, prob)
			if characteristic == "krnl_szs":
				mutate = self.mutate_krnl_szs(mutate, prob)
			if characteristic == "cnv_strd_szs":
				mutate = self.mutate_cnv_strd_szs(mutate, prob)	
	
	# In this function two CNN Individuals will be crossed
	# if mutate all hparams is true all the chromosomes will be crossed,
	# otherwise just a random chromosome
	def func_mate_CNN(self, ind1, ind2, mate_all_hparams=False):
		# print("Crossing took place")
		hddn_fc_lyrs_drpt_act_fncs_crossed = False
		cnv_lyrs_krnl_szs_cnv_strd_szs_crossed = False
		if mate_all_hparams:
			# print("Deberia entrar aqui solo una vez")
			for i in self.descriptor.evol_hparams:
				if (i == "hddn_fc_lyrs" or i == "drpt" or i == "act_fncs") and not hddn_fc_lyrs_drpt_act_fncs_crossed:
					ind1, ind2, hddn_fc_lyrs_drpt_act_fncs_crossed = self.mate_hddn_fc_lyrs_drpt_act_fncs(ind1, ind2, hddn_fc_lyrs_drpt_act_fncs_crossed)					
				if i == "lr":
					ind1, ind2 = self.mate_lr(ind1, ind2)
				if i == "epcs":
					ind1, ind2 = self.mate_epcs(ind1, ind2)
				if i == "pat":
					ind1, ind2 = self.mate_pat(ind1, ind2)
				if i == "hddn_fc_lyrs_sz":
					ind1, ind2 = self.mate_hddn_fc_lyrs_sz(ind1, ind2)
				if (i == "cnv_lyrs" or i == "krnl_szs" or i == "cnv_strd_szs") and not cnv_lyrs_krnl_szs_cnv_strd_szs_crossed:
					ind1, ind2, cnv_lyrs_krnl_szs_cnv_strd_szs_crossed = self.mate_cnv_lyrs_krnl_szs_cnv_strd_szs(ind1, ind2, cnv_lyrs_krnl_szs_cnv_strd_szs_crossed)
				if i == "cnv_lyrs_sz":
					ind1, ind2 = self.mate_cnv_lyrs_sz(ind1, ind2)
		else:
			characteristic = random.choice(self.descriptor.evol_hparams)
			if (characteristic == "hddn_fc_lyrs" or characteristic == "drpt" or characteristic == "act_fncs") and not hddn_fc_lyrs_drpt_act_fncs_crossed:
				ind1, ind2, hddn_fc_lyrs_drpt_act_fncs_crossed = self.mate_hddn_fc_lyrs_drpt_act_fncs(ind1, ind2, hddn_fc_lyrs_drpt_act_fncs_crossed)					
			if characteristic == "lr":
				ind1, ind2 = self.mate_lr(ind1, ind2)
			if characteristic == "epcs":
				ind1, ind2 = self.mate_epcs(ind1, ind2)
			if characteristic == "pat":
				ind1, ind2 = self.mate_pat(ind1, ind2)
			if characteristic == "hddn_fc_lyrs_sz":
				ind1, ind2 = self.mate_hddn_fc_lyrs_sz(ind1, ind2)
			if (characteristic == "cnv_lyrs" or characteristic == "krnl_szs" or characteristic == "cnv_strd_szs") and not cnv_lyrs_krnl_szs_cnv_strd_szs_crossed:
				ind1, ind2, cnv_lyrs_krnl_szs_cnv_strd_szs_crossed = self.mate_cnv_lyrs_krnl_szs_cnv_strd_szs(ind1, ind2, cnv_lyrs_krnl_szs_cnv_strd_szs_crossed)
			if characteristic == "cnv_lyrs_sz":
				ind1, ind2 = self.mate_cnv_lyrs_sz(ind1, ind2)

	# In this function the CNN network is initialized and returned
	def initialize_NN(self, descriptor_CNN):
		network = CNN_Network(descriptor_CNN)
		return network
			
		
	
	# In this function the CNN network is trained and validated
	def execute_NN(self, network, trainloader,validloader):
		network.training_CNN(trainloader)
		return network.testing_CNN(validloader)

	# ==== END NN functions ====


	# ==== BEGIN fitness, mutation and mate functions ====
	# In this function the CNN network's fitness is obtained
	def func_fitness(self, individual, trainloader, validloader):
		hidden_fc_layers = individual.hidden_fc_layers
		num_of_inputs = self.descriptor.nn_hparameters.input_dim
		num_of_outputs = self.descriptor.nn_hparameters.output_dim
		act_functions_ref = individual.act_functions_ref#self.descriptor.nn_hparameters.act_functions_ref
		dropout = individual.dropout#self.descriptor.nn_hparameters.dropout
		batch_size = self.descriptor.nn_hparameters.batch_size
		epochs = individual.epochs#self.descriptor.nn_hparameters.epochs
		learning_rate = individual.learning_rate#self.descriptor.nn_hparameters.learning_rate
		optim_ref = self.descriptor.nn_hparameters.optim_ref
		criter_ref = self.descriptor.nn_hparameters.criter_ref
		patience = individual.patience #self.descriptor.nn_hparameters.patience
		conv_layers = individual.conv_layers
		kernel_sizes = individual.kernel_sizes
		conv_stride_sizes = individual.conv_stride_sizes

		# If it gets executed properly, it will return the accuracy from the training + validation along with
		# the network
		try:
			network = self.initialize_NN(CNN_Descriptor(hidden_fc_layers, num_of_inputs , num_of_outputs , \
			act_functions_ref,dropout, batch_size, epochs, learning_rate , optim_ref, criter_ref, \
			self.descriptor.nn_hparameters.print_every, patience, conv_layers, kernel_sizes, conv_stride_sizes))
			# We try to execute the CNN network, if it fails it will return -Infinite, a value which makes it impossible for
			# the failed network to succeed as the best among the offspring
			ftnss = self.execute_NN(network, trainloader,validloader)
			return ftnss, network
		# If an error occurs, it will return -Infinity along with the default CNN
		except Exception as e:
			return float("-Inf"), CNN_Network(CNN_Descriptor())
		

	def mate_offspring_CNN(self, offspring):
		# We mate the individuals
		for child1, child2 in zip(offspring[::2], offspring[1::2]):
			# print("\n\n==============")
			# print("Children antes \n", child1, "\n||||||||\n", child2)
			# print("-------------")
			if random.random() < self.descriptor.mate_happen_prob:
				self.toolbox.mate_CNN(child1,child2)
				# print("Children despues \n", child1, "\n||||||||\n",child2)
				# print("==============\n\n")
				del child1.fitness.values
				del child2.fitness.values
		return offspring

	def mutate_offspring_CNN(self, offspring):
		for mutant in offspring:
			# print("\n\n==============")
			# print("Individual antes\n", mutant)
			# print("-------------")
			if random.random() < self.descriptor.mutation_happen_prob:
				self.toolbox.mutate_CNN(mutate=mutant)
				# print("Individual despues\n", mutant)
				# print("==============\n\n")
				del mutant.fitness.values
		return offspring

	# In this function we execute the genetic algorithm
	def simple_genetic_algorithm(self,trainloader,validloader, pic_name=None, pic_directory="./"):
		best_fitness = float("-Inf")
		best_ind = None

		best_fitnesses = []
		mean_fitnesses = []
		times = []

		pop = self.toolbox.population_CNN()
		fitnesses = []
		
		fitnesses = self.get_fitnesses(pop, trainloader, validloader)

		pop = self.apply_fitnesses(pop, fitnesses)
		# We get the accuracies of all the individuals within the population
		fits = [ind.fitness.values[0] for ind in pop]
		g = 0
		best_ind, best_fitness, halt = self.get_best_ind_fitness(fits, pop, best_fitness, best_ind)	

		while g < self.descriptor.generation_numb and not halt:
			# Create a new generation
			start_time = time.time()
			g += 1
			# print(" -- Generation %i --" %g)
			# Select individuals for the new generation
			offspring = self.toolbox.select(pop,len(pop))
			# Clonate individuals for the new generation
			offspring = list(map(self.toolbox.clone,offspring))
			# Apply crossover and mutate the new generation
			# a[start:end:step]
			# offspring[::2] from the first element to the last going by 2 units at a time
			# offspring[1::2] from the second element to the last going by 2 units at a time

			# We mate the individuals
			offspring = self.mate_offspring_CNN(offspring)
			# We mutate the individuals
			offspring = self.mutate_offspring_CNN(offspring)

			# We evaluate the individuals with an invalid fitness, which means they have been modified
			# If we have deleted its fitness the value of ind.fitness stops being valid
		
			invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

			fitnesses = []
			# We get the new fitnesses for the modified individuals
			fitnesses = self.get_fitnesses(invalid_ind, trainloader, validloader)
			# We get the new fitnesses for the modified individuals
			invalid_ind = self.apply_fitnesses(invalid_ind, fitnesses)
			
			pop_and_offspring = pop + offspring

			# pop[:] = offspring
			pop[:] = self.toolbox.select_best(pop_and_offspring)

			fits = [ind.fitness.values[0] for ind in pop]

			best_fitnesses.append(max(fits))
			
			mean_fitnesses.append(mean(fits))
			times.append(time.time()-start_time)

			# print("FITS ON THE {}th GENERATION: {}".format(g,fits))
			if len(list(fits)) > 0:
				best_ind, best_fitness, halt = self.get_best_ind_fitness(fits, pop, best_fitness, best_ind)	
		
		if pic_name == None:
			nn_and_pic_name = self.get_current_date_time_str()
		else:
			nn_and_pic_name = pic_name
		# self.save_graphic([i for i in range(g)], best_fitnesses, "Evolution of the best fitness","Generation", "Best fitness", nn_and_pic_name , pic_directory)
		self.save_NN_and_info(best_ind, best_fitness, nn_and_pic_name, pic_directory)
		return [i for i in range(g)], best_fitnesses, mean_fitnesses, times