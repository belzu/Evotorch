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


def show_graphic2(fitnesses, xlabel, ylabel):	
	plt.clf()
	# data to plot
	n_groups = 6
	# create plot
	fig, ax = plt.subplots()
	index = np.arange(n_groups)
	bar_width = 0.35
	opacity = 0.8

	rms = [i[0] for i in fitnesses]

	evols = [i[1] for i in fitnesses]

	plt.bar(index, rms, bar_width, alpha=opacity, color='b', label='Red. Mans', hatch="x")

	plt.bar(index + bar_width, evols, bar_width, alpha=opacity, color='r', label='Red. Evols', hatch="O")

	# tick_marks = np.arange(10)
	# plt.xticks(tick_marks, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], rotation=45, size = 10)

	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.xticks(index + bar_width, ('C.D. 1', 'C.D. 2', 'C.D. 3', 'C.D. 4', 'C.D. 5', 'C.D. 6'))
	plt.legend()
	plt.tight_layout()
	plt.show()

def save_graphic2(fitnesses, xlabel, ylabel, img_name = None, pic_directory = "./"):
	plt.clf()
	# data to plot
	# n_groups = 6 #MLP
	n_groups = 3 #CNN
	# create plot
	fig, ax = plt.subplots()
	index = np.arange(n_groups)
	bar_width = 0.35
	opacity = 0.8

	rms = [i[0] for i in fitnesses]

	evols = [i[1] for i in fitnesses]

	plt.bar(index, rms, bar_width, alpha=opacity, color='b', label='Red. Mans', hatch="x")

	plt.bar(index + bar_width, evols, bar_width, alpha=opacity, color='r', label='Red. Evols', hatch="O")
	
	plt.xlabel(xlabel, fontsize=20)
	plt.ylabel(ylabel, fontsize=20)
	# plt.xticks(index + bar_width, ('1', '2', '3', '4', '5', '6')) #MLP
	plt.xticks(index + bar_width, ('1', '2', '3')) #MLP
	plt.legend(loc='best', fontsize=15)
	plt.tick_params(axis='both', which='major', labelsize=10)
	plt.tight_layout()
	plt.savefig(pic_directory+img_name)

sns.set(style="darkgrid")
# plt.style.use('classic')
problem = int(sys.argv[1])
evol_bf = []
evol_mf = []
rm_bf = []
rm_mf = []

files_names = {1:"Mushroom", 2:"Dermatology", 3:"Forest_type", 4:"AirQuality", 5:"Forest_fire", 6:"Istanbul",7:"MNIST", 8:"Fashion_MNIST"}

# legends = ["Algor. Evol. {}".format(i) for i in range(1,7)] #MLP
legends = ["Algor. Evol. {}".format(i) for i in range(1,4)] #MLP

# for evol in range(1,7):
# 	best_fitnesses = [0 for i in range(30)]
# 	mean_fitnesses = [0 for i in range(30)]
# 	for exe in range(1,6):

# 		file = "Results_evolutions/{}/EV_{}_exec_{}.csv".format(files_names[problem], evol, exe)

# 		data_pd = pd.read_csv(file, sep=',')

# 		data_np = np.array(data_pd.values, dtype='float64')

# 		for i, bf in enumerate(data_pd["Best fitnesses"]):
# 			best_fitnesses[i] += float(bf) 

# 		for i, mf in enumerate(data_pd["Mean fitnesses"]):
# 			mean_fitnesses[i] += float(mf)


# 	for i in range(30):
# 		best_fitnesses[i] = best_fitnesses[i]/5
# 		mean_fitnesses[i] = mean_fitnesses[i]/5

# 	if problem == 1 or problem == 2 or problem == 3:
# 		evol_bf.append(max(best_fitnesses))
# 	else:
# 		evol_bf.append(min(best_fitnesses))
	
# 	evol_mf.append(max(mean_fitnesses))
# for evol in range(1,7):
for evol in range(1,4):

	best_fitnesses = []

	for exe in range(1,6):

		file = "Results_Redes_Evolucionadas/{}/REV_{}_exec_{}.csv".format(files_names[problem], evol, exe)

		data_pd = pd.read_csv(file, sep=',')

		data_np = np.array(data_pd.values, dtype='float64')

		best_fitnesses.append(data_pd["Fitness_test"][0])

	evol_bf.append(mean(best_fitnesses))

# for rm in range(1,7):
for rm in range(1,4):
	best_fitnesses = []
	for exe in range(1,6):

		file = "Results_Redes_Manuales/{}/RM_{}_exec_{}.csv".format(files_names[problem], rm, exe)

		data_pd = pd.read_csv(file, sep=',')

		data_np = np.array(data_pd.values, dtype='float64')

		best_fitnesses.append(data_pd["Fitness_test"][0])

	rm_bf.append(mean(best_fitnesses))

fitnesses = []
# for i in range(6): #MLP
for i in range(3): #CNN
	fitnesses.append([rm_bf[i], evol_bf[i]])


if problem == 1 or problem == 2 or problem == 3 or problem == 7 or problem == 8:
	save_graphic2(fitnesses, "Redes manuales y evolucionadas", "Precisión", files_names[problem]+"_rm_comp" ,"./Experiment_figs/")
else:
	save_graphic2(fitnesses, "Redes manuales y evolucionadas", "MSE", files_names[problem]+"_rm_comp" ,"./Experiment_figs/")