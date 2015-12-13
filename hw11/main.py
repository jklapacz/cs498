import infrastructure as inf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import plotly.plotly as py
import plotly.graph_objs as go

from sys import argv
from time import time
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans



__author__ = 'Jakub Klapacz <jklapac2@illinois.edu>'

debug = True


''' K-Means Parameters '''
num_clusters = 20 		# Number of clusters
pca_f = 1 				# Use num_cluster principal components as 
if pca_f == 1:			#	Initial cluster centers vs. random 
	num_init = 1		# Number of K-means trials
else:					#	The best run is used as
	num_init = 25		#	The estimator
num_iterations = 350	# Number of iterations per trial
k_size = 32
''' END PARAMETERS'''

category_z = defaultdict(list)

all_z = list()
labels = list()

X = list()
y = list()

def vectorize(array, category, size = k_size):
	# limit = len(array) / size
	limit = len(array) - (len(array) % size)

	cur = list()
	for i in range(0, limit, 16):
		# cur_idx = i*size
		cur_idx = i
		if(cur_idx + size >= limit):
			break
		curr = array[cur_idx:cur_idx + size, :]
		curr = np.ravel(curr)
		all_z.append(curr)
		labels.append(category)
		cur.append(curr)
	category_z[category].append(cur)	

def kmeans_setup(data):
	

	if pca_f == 1:
		pca = PCA(n_components = num_clusters).fit(data)
		initializer = pca.components_
		name = 'PCA'
	else:
		initializer = 'k-means++'
		name = 'k-means++'

	t0 = time()
	
	estimator = KMeans(init=initializer, n_clusters=num_clusters, n_init = num_init, max_iter = num_iterations)
	estimator.fit(data)
	
	if debug == True:
		sample_size = 300
		print('% 9s   %.2fs    %i   %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
	          % (name, (time() - t0), estimator.inertia_,
	             metrics.homogeneity_score(labels, estimator.labels_),
	             metrics.completeness_score(labels, estimator.labels_),
	             metrics.v_measure_score(labels, estimator.labels_),
	             metrics.adjusted_rand_score(labels, estimator.labels_),
	             metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
	             metrics.silhouette_score(data, estimator.labels_,
	                                      metric='euclidean',
	                                      sample_size=sample_size)))
	return estimator


def histograms(category, estimator):
	hist_num = 0
	make_hist = False
	for trial in category_z[category]:
		if hist_num == 3:
			make_hist = False
		
		current_data = np.asarray(trial)
		# limit = len(current_data) - (len(current_data) % k_size)

		# for i in range(0, limit, 16):
		# 	if()

		V = np.zeros(num_clusters)
		predicted = estimator.predict(current_data)
		total = 0
		
		for i in range(len(predicted)):
			total += 1
			V[predicted[i]] += 1
		for i in range(len(V)):
			V[i] = (V[i] * 1.0) / (total * 1.0)
			
		X.append(V)
		y.append(category)
		# print X
		# print y
		
		# print predicted
		# print V
		hist_num += 1
		if make_hist == True:
			filename_ = 'Histograms/' + category + str(hist_num) + '.png'
			# plt.hist(V, 20, normed=1)
			# plt.show()
			# quit()
			
			data = [
				go.Histogram(
					x=V
				)
			]
			title_ = category + ' ' + str(hist_num)
			layout = go.Layout(title=title_,
				xaxis = dict(
					title = "Cluster Center"),
				yaxis = dict(
					title = 'Frequency'))
			fig = go.Figure(data=data, layout=layout)
			py.image.save_as(fig, filename=filename_)
		else:
			continue


def main(load = 0):
	if load == 0:
		all_arrays = inf.traverse_files()
		pickle.dump(all_arrays, open("arrays.data", "wb"))
	else:
		all_arrays = pickle.load(open("arrays.data", "rb"))
	for entry in all_arrays:
		# print entry

		# print len(all_arrays[entry]) #This corresponds to how many files are in this category
		for i in all_arrays[entry]: #This is a particular file in a category
			arr = np.asarray(i)
			# print arr.shape
			vectorize(arr, entry)

	data = np.asarray(all_z)
	print data.shape
	estimator = kmeans_setup(data)
	for category in category_z:
		histograms(category, estimator)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=212)
	


	classifier = RandomForestClassifier()
	classifier.fit(X_train, y_train)
	classifier_predictions = classifier.predict(X_test)
	# classifier_predictions = classifier.predict(X)

	score = accuracy_score(y_test, classifier_predictions)
	# score = accuracy_score(y, classifier_predictions)
	print "Accuracy of Random Forest Classifier: ", score
	matrix = confusion_matrix(y_test, classifier_predictions)
	# matrix = confusion_matrix(y, classifier_predictions)
	
	outstr = '\n\t'
	for i in range(14):
		outstr+= '\t[' + str(i) + ']'
	for i in range(len(matrix)):
		outstr+= '\n\t[' + str(i) + ']'
		for j in range(len(matrix[0])):
			outstr+= '\t' + str(matrix[i][j])
	outstr += '\n'
	print outstr
	print matrix.shape
	print 'y test'
	print len(y_test)
	print 'y train'
	print len(y_train)
	print 'y'
	print len(y)
	# print len(np.unique(y))
	print len(np.unique(y_train))
	print np.unique(y_train)
	print np.unique(y_test)
	print len(np.unique(y_test))
	# print np.unique(y_test) 
	# print classifier_predictions







	

	



if __name__ == '__main__':
	load = 0
	if len(argv) > 1:
		if(argv[1] == '-l'):
			load = 1
	main(load)