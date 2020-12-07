import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import stdev as stdev
from statistics import mean as mean
from sklearn.preprocessing import MinMaxScaler
import warnings,os,sys
from datetime import datetime

def get_current_date_time_str():
	now = datetime.now()
	dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
	dt_string = dt_string.replace("/","-")
	dt_string = dt_string.replace(" ","_")
	dt_string = dt_string.replace(":","_")
	return str(dt_string)


def show_graphic(ax_x, ax_y, title, xlabel, ylabel):
	plt.clf()
	plt.xlim([min(ax_x), max(ax_x)])
	plt.ylim([min(ax_y), max(ax_y)])
	plt.title(title)
	plt.scatter(ax_x, ax_y)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.legend()
	plt.show()


def show_graphic2(ax_x, ax_y, title, xlabel, ylabel, legends):	
	plt.clf()
	plt.xlim([0, 29])
	# plt.ylim([min(min(ax_y))-0.115, max(max(ax_y)) +0.115 ])
	# plt.ylim([0, 1.5])
	# plt.ylim([min(min(ax_y)), 1.0])
	markers = {0:'o', 1:'x', 2:'D', 3:'*', 4:'>', 5:'p', 6:'s'}
	# plt.title(title)
	for idx,i in enumerate(ax_y):
		plt.plot([j for j in range(len(i))], i, marker = markers[idx])
	plt.margins(0.05)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.legend(legends, loc='best', prop={'size': 6}) # loc='upper left'bbox_to_anchor=(0.1, 0.95)
	plt.show()


def save_graphic2(ax_x, ax_y, title, xlabel, ylabel, legends, img_name = None, pic_directory = "./"):
	plt.clf()
	plt.xlim([0, 29])
	plt.ylim([min(min(ax_y))-0.005, max(max(ax_y)) +0.005 ])
	plt.margins(0.05)
	# plt.ylim([0.0, 1.05])
	# plt.ylim([min(min(ax_y)), 1.0])

	# markers = {0:'o',1:'x',2:'D',3:'*',4:'>',5:'p',6:'s'}#MLP
	markers = {0:'o',1:'x',2:'D'}#CNN
	# plt.title(title)
	for idx,i in enumerate(ax_y):
		plt.plot([j for j in range(len(i))], i, marker = markers[idx])
	
	plt.tick_params(labelsize=13)
	plt.xlabel(xlabel, fontsize=15)
	plt.ylabel(ylabel, fontsize=15)
	plt.legend(legends, loc='best', prop={'size': 12}) # loc='upper left'bbox_to_anchor=(0.1, 0.95)
	plt.savefig(pic_directory+img_name)




def save_graphic(ax_x, ax_y, title, xlabel, ylabel, img_name = None, pic_directory = "./"):
	plt.clf()
	plt.xlim([min(ax_x), max(ax_x)])
	plt.ylim([min(ax_y), (max(ax_y)+1.0)])
	if img_name == None:
		img_name = self.get_current_date_time_str()
	plt.title(title)
	# plt.scatter(torch.max(i).item(),labels[idx].item(),label="Result",alpha=0.5)
	plt.scatter(ax_x, ax_y)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.legend()
	plt.savefig(pic_directory+img_name)


sns.set(style="darkgrid")

problem = int(sys.argv[1])
evol_bf = []
evol_mf = []

files_names = {1:"Mushroom", 2:"Dermatology", 3:"Forest_type", 4:"AirQuality", 5:"Forest_fire", 6:"Istanbul",7:"MNIST", 8:"Fashion_MNIST"}

# legends = ["Algor. Evol. {}".format(i) for i in range(1,7)] #MLP
legends = ["Algor. Evol. {}".format(i) for i in range(1,4)] #CNN
# for evol in range(1,7): #MLP
for evol in range(1,4): #CNN
	best_fitnesses = [0 for i in range(30)]
	mean_fitnesses = [0 for i in range(30)]
	for exe in range(1,6):

		file = "Results_evolutions/{}/EV_{}_exec_{}.csv".format(files_names[problem], evol, exe)

		data_pd = pd.read_csv(file, sep=',')

		data_np = np.array(data_pd.values, dtype='float64')

		for i, bf in enumerate(data_pd["Best fitnesses"]):
			best_fitnesses[i] += float(bf) 

		for i, mf in enumerate(data_pd["Mean fitnesses"]):
			mean_fitnesses[i] += float(mf)


	for i in range(30):
		best_fitnesses[i] = best_fitnesses[i]/5
		mean_fitnesses[i] = mean_fitnesses[i]/5

	evol_bf.append(best_fitnesses)
	evol_mf.append(mean_fitnesses)

# print(best_fitnesses)


if problem == 1 or problem == 2 or problem == 3 or problem == 7 or problem == 8:
	save_graphic2([ [i for i in range(30)] for i in range(6)], evol_bf, "Método evolutivo {} para el\
		problema {}".format(evol,files_names[problem]), "Generaciones", "Precisión", legends, files_names[problem]+"_best" ,"./Experiment_figs/")
	save_graphic2([ [i for i in range(30)] for i in range(6)], evol_mf, "Método evolutivo {} para el\
		problema {}".format(evol,files_names[problem]), "Generaciones", "Precisión", legends, files_names[problem]+"_mean" ,"./Experiment_figs/")
else:
	save_graphic2([ [i for i in range(30)] for i in range(6)], evol_bf, "Método evolutivo {} para el\
		problema {}".format(evol,files_names[problem]), "Generaciones", "MSE", legends, files_names[problem]+"_best" ,"./Experiment_figs/")
	save_graphic2([ [i for i in range(30)] for i in range(6)], evol_mf, "Método evolutivo {} para el\
		problema {}".format(evol,files_names[problem]), "Generaciones", "MSE", legends, files_names[problem]+"_mean" ,"./Experiment_figs/")