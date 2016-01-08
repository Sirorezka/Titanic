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
	data = load_and_append (data,"output/prob_svm.csv")
	data = load_and_append (data,"output/prob_k_neighbors.csv")
	header = ["id","surv","adaboost","svm","k_neighbors"]


	print (header)
	print (data)

	x_id = np.copy(data[:,0])
	has_surv = (data[:,1]!="")
	y = np.copy(data[has_surv,1]).astype(int)
	X = np.copy(data[has_surv,2:]).astype(float)
	X_train, X_test, y_train, y_test = train_test_split(
						X, y, train_size=0.7, random_state=13)

	print ("surv all: ",sum(y==1)/len(y))
	print ("surv train: ",sum(y_train==1)/len(y_train))
	print ("surv test: ",sum(y_test==1)/len(y_test))



	# MONTE CARLO WEIGHT SEARCH
	random.seed()
	all_sol = []
	for k in range(9500):
		w_weights = [1,1,1]
		w_int = [0] + w_weights[:-1] + [1]

		for j in range(1,len(w_int)-1):
			w_int[j] = random.random()
		w_int = np.sort(w_int)
		for j in range(len(w_weights)):
			w_weights[j] = w_int[j+1]-w_int[j]

		#print (w_weights)

		model_av = np.average(X_train,axis=1, weights=w_weights)
		y_prediction = model_av>=0.5
		y_score_train = round(sum(y_prediction==y_train)/len(y_train),3)
		y_surv_train = round(sum(y_prediction==1)/len(y_train),3)

		model_av = np.average(X_test,axis=1, weights=w_weights)
		y_prediction = model_av>=0.5
		y_score_test = round(sum(y_prediction==y_test)/len(y_test),3)
		y_surv_test = round(sum(y_prediction==1)/len(y_test),3)
		w_weights = list(map(lambda x: round(x,3),w_weights))

		d_new = [0.5, y_score_train, y_surv_train, y_score_test,y_surv_test] + w_weights
		all_sol.append(d_new)
		#print (0.5, y_score,y_surv)


	all_sol = np.array(all_sol)
	ind_sort = np.argsort (all_sol[:,1])[::-1][:10]
	# printing best weights
	print ("Best train")
	print (all_sol[ind_sort])
	ind_sort = np.argsort (all_sol[:,3])[::-1][:10]
	# printing best weights
	print ("Best test")
	print (all_sol[ind_sort])
	all_sol = all_sol[ind_sort]

	print (header[2:8])
	print ("Best solution: ", all_sol[0,:])
	w_weights = all_sol[0,5:8]
	print ("Best weights: ", w_weights)

	# searching for best probability
	# prob_test = np.average(X_test,axis=1, weights=w_weights)
	# i_rng = np.linspace(0,1,25)
	# for i in i_rng:
	# 	score_true = sum((prob_test>=i).astype(int)==y_test)/len(y_test)
	# 	print (round(i,2), score_true)

	x_id = np.copy(data[~has_surv,0])
	X_nolabel = data[~has_surv,2:].astype(float)
	model_av = np.average(X_nolabel,axis=1, weights=w_weights)
	y_predict = model_av>=0.5
	y_predict = y_predict.astype(int)
	prob_surv = round(sum(y_predict==1)/len(y_predict),4)
	print ("predicted data: ", prob_surv)

	predictions_file = open("output/stacking2_weigted_av.csv", "w", newline='')

	predictions_file_object = csv.writer(predictions_file)
	predictions_file_object.writerow(["PassengerId", "Survived"])	
	#predictions_file_object.writerow([pass_id, y_predict])

	for i in range(len(y_predict)):														
		predictions_file_object.writerow([x_id[i], y_predict[i]])	

if __name__ == "__main__":
	main()