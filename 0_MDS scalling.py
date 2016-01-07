""" This simple code is desinged to teach a basic user to read in the files in python, simply find what proportion of males and females survived and make a predictive model based on this
Author : AstroDave
Date : 18 September 2012
Revised: 28 March 2014

"""

from multiprocessing import Pool
import matplotlib.pyplot as plt

import csv as csv
import numpy as np
import graphviz as gv
from sklearn import tree, svm,preprocessing
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn import ensemble 
from my_functions import *
from sklearn.manifold import MDS
from sklearn.decomposition import PCA

def main ():

	## LOADING DATA

	my_project_dir = "kaggle_data\\"
	csv_file_object = csv.reader(open(my_project_dir + 'processed.csv', 'r')) 	# Load in the csv file
	header = next(csv_file_object) 						# Skip the fist line as it is a header
	data=[] 												# Create a variable to hold the data

	for row in csv_file_object: 							# Skip through each row in the csv file,
		data.append(row[0:]) 								# adding each row to the data variable


	data = np.array(data) 									# Then convert from a list to an array.


	cols_for_model = ["PassengerId", "Survived","Pclass","Mrs","Mr","Miss","Master",
						"Captain","Other","is_female","Age_predict","Age_unknown",
						"age_less_10","age_10_20","age_20_30","age_30_50","age_more_50",
						"SibSp","Parch","Fare","No_cabin","Embarked_C","Embarked_Q",
						"Embarked_S","Same_Ticket"
						,"Same_Room_Surv","Same_Room_surv_perc"
						]

	col_remove = []
	for i in range(len(header)):
		if (header[i] not in cols_for_model): col_remove.append(i)

	print ("Zero value test: ", len(cols_for_model)-(len(header)-len(col_remove)))

	data = np.delete(data, col_remove,1)
	header = np.delete(header,col_remove,0)

	## Scaling data
	print ("Scaling features: ", header[[10,19]])
	data[:,[10,19]] = preprocessing.scale(data[:,[10,19]].astype(float))


	data_other = np.copy(data[:,2:])
	# clf_map = MDS()
	# my_map = clf_map.fit(data_other).embedding_
	# # my_map = data[:,4:6]
	# # print (my_map.shape)

	##
	## http://scikit-learn.org/stable/auto_examples/manifold/plot_mds.html
	##

	data_other = data_other.astype(float)
	pca = PCA(n_components=2)
	pca.fit(data_other) 
	my_map=pca.transform(data_other)
	print (my_map)


	y = data[:,1]
	y [y==""] = 2
	y = y.astype(int)
	cmap = plt.get_cmap('gnuplot')
	y_col = list(map(lambda x: cmap(i), y))

	colors = ["red","blue","green","yellow"]
	for k in np.unique(y):
		plt.scatter(my_map[y==k,0],my_map[y==k,1],color=colors[k])
	plt.show()


if __name__ == "__main__":
	main()