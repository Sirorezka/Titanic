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



	### Collecting data for model
	has_surv = (data[:,1] !='')

	y = data[has_surv,1]
	X = data[has_surv,2::]
	feature_names = header[2:]

	n_estim_min = 0
	samples_leaf_min = 0
	score_min = 0
	n_estim_mean = 0
	samples_leaf_mean = 0
	score_mean = 0


	### Best params:
	###  100  19  min 0.8125
	###  50   16  mean 0.833859468234
	###
	# grid search result
	# 0.81981981982 70 11  - Min
	# 0.83387957763 60 14  - Mean
	# random_state  = 1013


	### Making prediction based on data sampled from train set
	# for i in range(10,25):
	# 	for j in range (20,300,10):
			#print (i,j)

	clf = ensemble.RandomForestClassifier(n_estimators=70, min_samples_leaf =11,
										  random_state  = 1013)
	clf = clf.fit(X,y)
	scores = cross_val_score(clf, X, y, cv = 8, n_jobs =5)
	print ("survived: ",sum (y=='1')/len(y))
	print (X.shape[0])
	print ("Crossvalidation scores: ")
	print (scores)
	print("min: ",scores.min(),"mean: ",scores.mean())

			# print ("\n")
			# if scores.min()>score_min:
			# 	score_min = scores.min()
			# 	n_estim_min = j
			# 	samples_leaf_min = i

			# if scores.mean()>score_mean:
			# 	score_mean = scores.mean()
			# 	n_estim_mean = j
			# 	samples_leaf_mean = i

	print ("grid search result")
	print (score_min, n_estim_min, samples_leaf_min)
	print (score_mean, n_estim_mean, samples_leaf_mean)


	print ("\n \n \n")
	#save_tree_img ("img/tree.dot", clf, feature_names, class_names =["dead","survived"])
	#print (clf.feature_importances_)

	importances = clf.feature_importances_
	indices = np.argsort(importances)[::-1]

	for f in range(len(feature_names)):
   		 print("%d. feature %s (%f)" % (f + 1, feature_names[f], importances[indices[f]]))


	# predicting data
	x_id = data[~has_surv,0]
	X = data[~has_surv,2::]
	y_predict = clf.predict(X)
	print ("\ntest predict")
	print ("survived: ",sum (y_predict=='1')/len(y_predict))


	predictions_file = open("output/random_forest.csv", "w", newline='')
	predictions_file_object = csv.writer(predictions_file)
	predictions_file_object.writerow(["PassengerId", "Survived"])	
	#predictions_file_object.writerow([pass_id, y_predict])

	for i in range(len(y_predict)):														
	    predictions_file_object.writerow([x_id[i], y_predict[i]])			
														
if __name__ == "__main__":
    main()