""" This simple code is desinged to teach a basic user to read in the files in python, simply find what proportion of males and females survived and make a predictive model based on this
Author : AstroDave
Date : 18 September 2012
Revised: 28 March 2014

"""


import csv as csv
import numpy as np
import graphviz as gv
from sklearn import tree
from sklearn.cross_validation import cross_val_score
from sklearn import ensemble 
from sklearn import linear_model
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


	# for i in range(len(header)):
	# 	print (i," ",header[i])


	col_remove = [3, 5, 8, 12,14,15,17,18]               # Column 5 represent age
	data = np.delete(data, col_remove,1)
	header = np.delete(header,col_remove,0)



	### Collecting data for model
	has_surv = (data[:,1] !='')
	#print (has_surv)
	n_par = int(sum(has_surv)*0.75) 
	
	# y = data[0:n_par,1]
	# X = data[0:n_par,2::]
	y = data[has_surv,1]
	X = data[has_surv,2::]
	feature_names = header[2:]


	### Making prediction based on data sampled from train set
	min = 0
	param_min = [0,0]
	mean = 0
	param_mean = [0,0]
	i_samples = 55
	i_estimators = 460

	clf = ensemble.AdaBoostClassifier(
	    tree.DecisionTreeClassifier(min_samples_leaf=100),
	    n_estimators=12,
	    algorithm="SAMME")

	clf = clf.fit(X, y)
	scores = cross_val_score(clf, X, y, cv = 8, n_jobs=-1)
	print ("adaboost")
	print ("survived: ",sum (y=='1')/len(y))
	print (X.shape[0])
	print ("Crossvalidation scores: ")
	print (scores)
	print("min: ",scores.min(),"   mean:",scores.mean())
	print ("\n")
	if scores.min()>min: param_min = [i_samples, i_estimators]
	if scores.mean()>mean: param_mean = [i_samples, i_estimators]


	print ("min: ", param_min)
	print ("mean: ", param_mean)



	y = data[n_par:sum(has_surv),1]
	X = data[n_par:sum(has_surv),2::]
	y_predict = clf.predict(X)
	print ("survived: ",sum (y=='1')/len(y))
	print("Blind prediction: ", sum(y_predict==y)/len(y))

	#save_tree_img ("img/tree.dot", clf, feature_names, class_names =["dead","survived"])


	# predicting data
	x_id = data[~has_surv,0]
	X = data[~has_surv,2::]
	y_predict = clf.predict(X)
	print ("\ntest predict")
	print ("survived: ",sum (y_predict=='1')/len(y_predict))


	predictions_file = open("output/adaboost.csv", "w", newline='')
	predictions_file_object = csv.writer(predictions_file)
	predictions_file_object.writerow(["PassengerId", "Survived"])	
	#predictions_file_object.writerow([pass_id, y_predict])

	for i in range(len(y_predict)):														
	    predictions_file_object.writerow([x_id[i], y_predict[i]])			
														
if __name__ == "__main__":
    main()