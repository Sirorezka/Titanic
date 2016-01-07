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
		arr = np.column_stack((arr,dr[:,1]))
	return arr



def main ():



	data = []
	data = load_and_append (data,"kaggle_data/processed.csv")
	data = data [:,0:2]

	data = load_and_append (data,"output/full_best_adaboost.csv")
	data = load_and_append (data,"output/full_best_decisiontree.csv")
	data = load_and_append (data,"output/full_best_random_forest.csv")
	data = load_and_append (data,"output/full_best_svm.csv")
	data = load_and_append (data,"output/full_best_xgboost.csv")
	data = load_and_append (data,"output/full_best_logistic.csv")
	header = ["id","surv","adaboost","decistree","rand_forest","svm","xgboost","logistic"]


	print (header)
	x_id = np.copy(data[:,0])
	has_surv = (data[:,1]!="")
	y = np.copy(data[has_surv,1]).astype(int)
	
	X = np.copy(data[has_surv,2:]).astype(float)

	X_train, X_test, y_train, y_test = train_test_split(
									X, y, test_size=0.5, random_state = 42)
	print ("surv all: ",sum(y==1)/len(y))
	print ("surv train: ",sum(y_train==1)/len(y_train))
	print ("surv test: ",sum(y_test==1)/len(y_test))


	i_norm = 'l1'
	i_c = 1
	c_range = np.arange(0.1,5,0.05)


	clf = LogisticRegression(penalty=i_norm,C = i_c, n_jobs = -1)
	scores = cross_val_score(clf, X_train, y_train, cv = 8, n_jobs=-1)
	#print (scores)
	clf.fit(X_train, y_train)
	y_predict = clf.predict(X_test)
	x_prob = clf.predict_proba(X_test)
	score_cv_min = round(scores.min(),4)
	score_cv_mean = round(scores.mean(),4)
	prob_surv = round(sum(y_predict==1)/len(y_test),4)
	score_acc = round(sum(y_predict==y_test)/len(y_test),4)
	score_f1 = round(f1_score(y_test,y_predict),4)
	print (prob_surv, score_acc, score_f1, scores.min(), scores.mean())
	#print (clf.coef_)


	clf.fit(X_train, y_train)
	x_id = np.copy(data[~has_surv,0])
	X_nolabel = data[~has_surv,2:].astype(float)
	y_predict = clf.predict(X_nolabel)	
	prob_surv = round(sum(y_predict==1)/len(y_test),4)
	print (prob_surv)

	predictions_file = open("output/stacking.csv", "w", newline='')

	predictions_file_object = csv.writer(predictions_file)
	predictions_file_object.writerow(["PassengerId", "Survived"])	
	#predictions_file_object.writerow([pass_id, y_predict])

	for i in range(len(y_predict)):														
		predictions_file_object.writerow([x_id[i], y_predict[i]])	


if __name__ == "__main__":
	main()