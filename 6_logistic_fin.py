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


	### Collecting data for model
	has_surv = (data[:,1] !='')

	y = np.copy(data[has_surv,1].astype(float))
	X = np.copy(data[has_surv,2::].astype(float))
	feature_names = header[2:]


	# dividing into test/control group
	X_train, X_test, y_train, y_test = train_test_split(
					X, y, test_size=0.3, random_state=444)

	print ("surv all: ",sum(y==1)/len(y))
	print ("surv train: ",sum(y_train==1)/len(y_train))
	print ("surv test: ",sum(y_test==1)/len(y_test))



	#c_grid = np.logspace(-1, 1, 20)

	i_c = 0.3   # optimal parameter
	i_norm = "l1"
	clf = LogisticRegression(penalty=i_norm,C = i_c, n_jobs = -1)
	scores = cross_val_score(clf, X, y, cv = 8, n_jobs=-1)
	print (scores)
	clf.fit(X_train, y_train)
	y_predict = clf.predict(X_test)
	x_prob = clf.predict_proba(X_test)
	score_cv_min = round(scores.min(),4)
	score_cv_mean = round(scores.mean(),4)
	score_acc = round(sum(y_predict==y_test)/len(y_test),4)
	score_f1 = round(f1_score(y_test,y_predict),4)
	
	print (i_c, score_cv_min, score_cv_mean, score_acc, score_f1)




	# testing optimal probability
	pr_grid = np.arange(0,1,0.05)
	print(pr_grid)

	## optimal prob = 0.55
	p_optim = 0.55
	clf.fit(X_train, y_train)
	print (X_test.shape)
	res_acc = []
	res_f1 = []
	for i_pr in pr_grid:
		y_predict = clf.predict(X_test)
		x_prob = clf.predict_proba(X_test)
		tt = np.column_stack((x_prob,y_predict))
		y_predict[:] = 0
		y_predict[x_prob[:,1]>=i_pr] = 1
		score_acc = round(sum(y_predict==y_test)/len(y_test),4)
		score_f1 = round(f1_score(y_test,y_predict),4)
		res_acc.append(score_acc)
		res_f1.append(score_f1)
		print (i_pr,sum(y_predict),score_acc, score_f1)

	plt.plot(pr_grid,res_acc)
	plt.plot(pr_grid,res_f1)
	plt.show()




	# predicting data

	#clf.fit(X, y)	
	X_nolabel = data[:,2::].astype(float)
	x_id = data[:,0]
	clf.fit(X_train, y_train)
	print (X_nolabel)
	y_predict = clf.predict(X_nolabel).astype(int)
	y_prob = clf.predict_proba(X_nolabel)
	y_predict[:]=0
	y_predict[y_prob[:,1]>=p_optim] = 1
	print ("dsf")
	print (y_predict)
	print ("\ntest predict")
	print ("survived: ",sum (y_predict==1)/len(y_predict))



	predictions_file = open("output/prob_logistic.csv", "w", newline='')

	predictions_file_object = csv.writer(predictions_file)
	predictions_file_object.writerow(["PassengerId", "Survived"])	
	#predictions_file_object.writerow([pass_id, y_predict])

	for i in range(len(y_predict)):														
		predictions_file_object.writerow([x_id[i], y_predict[i],y_prob[i,0],y_prob[i,1]])			
														
if __name__ == "__main__":
	main()