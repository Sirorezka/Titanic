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
	print (data)

	x_id = np.copy(data[:,0])
	has_surv = (data[:,1]!="")
	y = np.copy(data[has_surv,1]).astype(int)
	X = np.copy(data[has_surv,2:]).astype(float)
	X_train, X_test, y_train, y_test = train_test_split(
									X, y, test_size=0.5, random_state = 42)

	print ("surv all: ",sum(y==1)/len(y))
	print ("surv train: ",sum(y_train==1)/len(y_train))
	print ("surv test: ",sum(y_test==1)/len(y_test))



	# MONTE CARLO WEIGHT SEARCH
	random.seed()
	all_sol = []
	for k in range(9500):
		w_int = [0,1,1,1,1,1,1]
		w_weights = [1,1,1,1,1,1]
		for j in range(1,6):
			w_int[j] = random.random()
		w_int = np.sort(w_int)
		for j in range(6):
			w_weights[j] = w_int[j+1]-w_int[j]

		model_av = np.average(X_train,axis=1, weights=w_weights)
		y_prediction = model_av>=0.5
		y_score_train = sum(y_prediction==y_train)/len(y_train)
		y_surv_train = sum(y_prediction==1)/len(y_train)

		model_av = np.average(X_test,axis=1, weights=w_weights)
		y_prediction = model_av>=0.5
		y_score_test = sum(y_prediction==y_test)/len(y_test)
		y_surv_test = sum(y_prediction==1)/len(y_test)		

		d_new = [0.5, y_score_train, y_surv_train, y_score_test,y_surv_test] + w_weights
		all_sol.append(d_new)
		#print (0.5, y_score,y_surv)


	all_sol = np.array(all_sol)
	ind_sort = np.argsort (all_sol[:,1])[::-1][:25]
	# printing best weights
	print (all_sol[ind_sort])

	print (header[2:8])
	w_weights = all_sol[0,5:11]
	print (w_weights)
	x_id = np.copy(data[~has_surv,0])
	X_nolabel = data[~has_surv,2:].astype(float)
	model_av = np.average(X_nolabel,axis=1, weights=w_weights)
	y_predict = model_av>=0.5
	y_predict = y_predict.astype(int)
	prob_surv = round(sum(y_predict==1)/len(y_test),4)
	print ("predicted data: ", prob_surv)

	predictions_file = open("output/stacking_weights_prob_v1.csv", "w", newline='')

	predictions_file_object = csv.writer(predictions_file)
	predictions_file_object.writerow(["PassengerId", "Survived"])	
	#predictions_file_object.writerow([pass_id, y_predict])

	for i in range(len(y_predict)):														
		predictions_file_object.writerow([x_id[i], y_predict[i]])	

if __name__ == "__main__":
	main()