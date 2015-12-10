import numpy as np
from sklearn.preprocessing import StandardScaler
import math
from svm import *
import plotly.plotly as py
# import plotly.graph_objs as go
from plotly.graph_objs import *
data = np.loadtxt("wdbc.data", delimiter=",", dtype='object')

length = len(data)
limit = 100
keep = np.random.permutation(length)[:limit]
testdata = data[keep]

# newdata = np.delete(data, keep, 0)
trainingdata = np.delete(data, keep, 0)
# length = len(newdata)
# keep = np.random.permutation(length)[:limit]

# validationdata = newdata[keep]
# trainingdata = np.delete(newdata, keep, 0)



# Ignore first column, first column is just IDs
y = data[:, 1]
X = data[:, 2:].astype(float)
X = StandardScaler().fit_transform(X)  # Rescales data

testy = testdata[:, 1]
testX = testdata[:, 2:].astype(float)
testX = StandardScaler().fit_transform(testX)  # Rescales data

# validy = validationdata[:, 1]
# validX = validationdata[:, 2:].astype(float)
# validX = StandardScaler().fit_transform(validX)  # Rescales data

trainy = trainingdata[:, 1]
trainX = trainingdata[:, 2:].astype(float)
trainX = StandardScaler().fit_transform(trainX)  # Rescales data

# TODO: Train your SVM based upon different regularization constants
regs = [1e-3, 1e-2, 1e-1, 1]
scatter_plot_objs = list()
accuracies = list()
for reg in regs:
	curr = SVC(reg)
	result = curr.fit(trainX, trainy)
	# print accuracy
	
	
	
	obj = Scatter(x = result[0][0], y = result[1][0])
	scatter_plot_objs.append(obj)
	# validations.append(fit)
	total = 0
	correct = 0
	newy = np.zeros(len(testy))
	classes = np.unique(testy)
	newy[testy == classes[0]] = -1
	newy[testy == classes[1]] = 1
	for i in range(len(testX)):
		total += 1
		if (newy[i] == curr.predict(testX[i])):
			correct += 1
	accuracy = (correct * 1.0) / (total * 1.0)
	# print accuracy
	# quit()
	accuracies.append((reg, accuracy))
	# break
print accuracies
quit()
data = Data([scatter_plot_objs[0], scatter_plot_objs[1], scatter_plot_objs[2], scatter_plot_objs[3]])
layout = Layout(title='Fig 3: Accuracy every 10 steps (3rd iteration)',
	xaxis = dict(
		title = "Steps [10's]"),
	yaxis = dict(
		title = 'Accuracy'))
fig = Figure(data=data, layout=layout)
py.image.save_as(fig, filename='113.png')
# print data
# quit()

# stats = np.asarray(validations)
# stats = np.matrix(stats)
# data = [
# 	go.Scatter(
# 		x = stats[:, 1].tolist(),
# 		# x = np.reshape(stats[:, 1], (1, len(stats[:, 1]))),
# 		y = stats[:, 2].tolist()
# 		# y = np.reshape(stats[:, 2], (1, len(stats[:, 2])))
# 	)
# ]
# py.image.save_as({'data': data}, '112.png')
# print stats[:, 1].tolist()

# print np.reshape(stats[:, 1], (1, len(stats[:, 1]))