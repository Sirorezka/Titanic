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
									X, y, test_size=0.5)

	print ("surv all: ",sum(y==1)/len(y))
	print ("surv train: ",sum(y_train==1)/len(y_train))
	print ("surv test: ",sum(y_test==1)/len(y_test))

	clf = tree.DecisionTreeClassifier(min_samples_leaf=12)
	scores = cross_val_score(clf, X, y, cv = 8)
	print (X.shape[0])
	print ("Crossvalidation scores: ")
	print (scores)
	print("min: ",scores.min(),"  mean: ", scores.mean())
	clf.fit(X,y)
	y_prob = clf.predict(X_test)
	print (sum(y_prob==y_test)/len(y_test))

	save_tree_img ("img/stack_tree.dot", clf, header[2:8], class_names =["dead","survived"])


	x_id = np.copy(data[~has_surv,0])
	X_nolabel = data[~has_surv,2:].astype(float)
	y_predict = clf.predict(X_nolabel)
	y_predict = y_predict.astype(int)
	prob_surv = round(sum(y_predict==1)/len(y_test),4)
	print ("predicted data: ", prob_surv)

	predictions_file = open("output/stacking_tree.csv", "w", newline='')

	predictions_file_object = csv.writer(predictions_file)
	predictions_file_object.writerow(["PassengerId", "Survived"])	
	#predictions_file_object.writerow([pass_id, y_predict])

	for i in range(len(y_predict)):														
		predictions_file_object.writerow([x_id[i], y_predict[i]])	


if __name__ == "__main__":
	main()