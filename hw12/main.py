import numpy as np
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go


from sys import argv
from time import time

from collections import defaultdict
from sklearn import linear_model
# from sklearn import metrics
# from sklearn.metrics import confusion_matrix, accuracy_score
# from sklearn.cross_validation import train_test_split
# from sklearn.preprocessing import OneHotEncoder, LabelEncoder



def main():
	data = np.loadtxt("kittiwak.txt", delimiter="\t", dtype=str)
	data = np.asarray(data)
	
	area = data[:, 1]
	area = area.astype(float)
	population = data[:, 2]
	population = population.astype(float)
	log_area = np.log(area)
	log_population = np.log(population)
	print population.shape
	print area.shape
	population = population.reshape((-1, 1))
	area = area.reshape((-1, 1))
	log_area = log_area.reshape((-1, 1))
	log_population = log_population.reshape((-1, 1))
	print population.shape
	print area.shape
	print len(population)
	print len(area)
	combinations = [(area, population), (area, log_population), (log_area, log_population), (log_area, population)]
	labels = ["Population from Area", "log(Population) from Area", "log(Population) from log(Area)", "Population from log(Area)"]
	xlabels = ["Area", "Area", "log(Area)", "log(Area)"]
	ylabels = ["Population", "log(Population)", "log(Population)", "Population"]
	idx = 0
	for combination in combinations:
		a = combination[0]
		p = combination[1]
		filename = str(idx) + '.png'
		regr = linear_model.LinearRegression()	
		regr.fit(a, p)
		fig = plt.figure()
		ax = fig.add_subplot(111)
		plt.scatter(a, p, color='black')
		plt.plot(a, regr.predict(a), color='blue', linewidth=3)
		plt.xlabel(xlabels[idx])
		plt.ylabel(ylabels[idx])
		plt.title(labels[idx])
		txt = 'R^2: {0:.4f}'.format(regr.score(a, p))
		plt.text(.22, .9, txt, ha='center', va='center', transform=ax.transAxes, color="red")
		# plt.xticks(())
		# plt.yticks(())
		# plt.savefig(filename, bbox_inches='tight')
		plt.savefig(filename)
		plt.clf()
		idx += 1
	# regr.fit(area, population)
	
	# regr.fit(log_area, log_population)
	# regr.fit(log_area, population)
	
	
	
	
	


if __name__ == '__main__':
	t0 = time()
	main()
	t1 = time() - t0
	print "Time elapsed:\t{0:.4f}".format(t1) 
