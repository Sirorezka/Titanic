""" This simple code is desinged to teach a basic user to read in the files in python, simply find what proportion of males and females survived and make a predictive model based on this
Author : AstroDave
Date : 18 September 2012
Revised: 28 March 2014

"""

from multiprocessing import Pool
import matplotlib.pyplot as plt

import random
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



def load_and_append (arr, res_file):
	fo = open(res_file,"r")
	csv_fo =  csv.reader(fo)
	next(csv_fo)
	dr = []
	for line in csv_fo:
		dr.append(line)
	dr = np.array(dr)

	if arr ==[]:
		arr = dr
	else:
		arr = np.column_stack((arr,dr[:,3]))
	return arr



def main ():



	data = []
	data = load_and_append (data,"kaggle_data/processed.csv")


	data = data [:,0:2]

	data = load_and_append (data,"output/prob_adaboost.csv")
	data = load_and_append (data,"output/prob_decisiontree.csv")
	data = load_and_append (data,"output/prob_random_forest.csv")
	data = load_and_append (data,"output/prob_svm.csv")
	data = load_and_append (data,"output/prob_xgboost.csv")
	data = load_and_append (data,"output/prob_logistic.csv")
	header = ["id","surv","adaboost","decistree","rand_forest","svm","xgboost","logistic"]



	print (header)

	feature_names = header[2:]
	print (feature_names)
	x_id = np.copy(data[:,0])
	has_surv = (data[:,1]!="")
	y = np.copy(data[has_surv,1]).astype(int)
	X = np.copy(data[has_surv,2:]).astype(float)
	X_train, X_test, y_train, y_test = train_test_split(
									X[:,:], y, train_size=0.75, random_state=687)


	print ("surv all: ",sum(y==1)/len(y))
	print ("surv train: ",sum(y_train==1)/len(y_train))
	print ("surv test: ",sum(y_test==1)/len(y_test))


	probs = np.average(X, axis = 1)
	#print (probs)

	i_lsp = np.linspace (0,1,20)
	for i in i_lsp:
		y_predict = (probs>=i).astype(int)
		score_acc = sum (y_predict==y)/len(y)

		print (round(i,2),score_acc)



	X_no_label = data[~has_surv,2:].astype(float)
	probs = np.average(X_no_label, axis = 1)
	p_predict = (probs>=0.5).astype(int)
	x_id = data[~has_surv,0]

	perc_surv = round(sum(p_predict==1)/len(p_predict),4)
	print ("survived: ", perc_surv)


	predictions_file = open("output/stacking_prob_all.csv", "w", newline='')

	predictions_file_object = csv.writer(predictions_file)
	predictions_file_object.writerow(["PassengerId", "Survived"])	
	#predictions_file_object.writerow([pass_id, y_predict])

	for i in range(len(p_predict)):														
		predictions_file_object.writerow([x_id[i], p_predict[i]])	


if __name__ == "__main__":
	main()