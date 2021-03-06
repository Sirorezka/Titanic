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



	# Logistic regression
	i_norm = 'l2'


	c_range = np.linspace (0.01,15,45)

	# for i_c in c_range:
	# 	clf = LogisticRegression(penalty=i_norm,C = i_c, n_jobs = -1)
	# 	scores = cross_val_score(clf, X_train, y_train, cv = 8, n_jobs=-1)
	# 	clf.fit(X_train,y_train)
	# 	y_predict = clf.predict(X_test)
	# 	x_prob = clf.predict_proba(X_test)
	# 	score_cv_min = round(scores.min(),4)
	# 	score_cv_mean = round(scores.mean(),4)
	# 	score_acc = round(sum(y_predict==y_test)/len(y_test),4)
	# 	score_f1 = round(f1_score(y_test,y_predict),4)
		
	# 	#print (scores)
	# 	print ("Results: ",round(i_c,2), score_cv_min, score_cv_mean, score_acc, score_f1)


	### ALSO YOU CAN TRY TO USE 10!!!
	i_c = 4.1
	clf = LogisticRegression(penalty=i_norm,C = i_c, n_jobs = -1)
	scores = cross_val_score(clf, X_train, y_train, cv = 8, n_jobs=-1)
	print (scores)
	clf.fit(X_train,y_train)


	## Test score 0.8208 when prob = 0.57
	i_rn = np.linspace (0,1,20)
	y_pred = clf.predict_proba (X_test)[:,1]
	for i in i_rn:
		c_score = sum((y_pred>=i).astype(int)==y_test)/len(y_test)
	 	print (i, c_score)


	x_id = np.copy(data[~has_surv,0])
	X_nolabel = data[~has_surv,2:].astype(float)
	print (X_nolabel.shape)



	#y_predict = clf.predict(X_nolabel).astype(int)
	y_proba =  (clf.predict_proba(X_nolabel)[:,1]>=0.57).astype(int)
	y_predict = y_proba

	prob_surv = round(sum(y_predict==1)/len(y_predict),4)
	print ("predicted data: ", prob_surv)

	predictions_file = open("output/stacking2_logistic.csv", "w", newline='')

	predictions_file_object = csv.writer(predictions_file)
	predictions_file_object.writerow(["PassengerId", "Survived"])	
	#predictions_file_object.writerow([pass_id, y_predict])

	for i in range(len(y_predict)):														
		predictions_file_object.writerow([x_id[i], y_predict[i]])	

if __name__ == "__main__":
	main()