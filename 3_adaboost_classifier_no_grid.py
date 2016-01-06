""" This simple code is desinged to teach a basic user to read in the files in python, simply find what proportion of males and females survived and make a predictive model based on this
Author : AstroDave
Date : 18 September 2012
Revised: 28 March 2014

"""


import csv as csv
import numpy as np
import graphviz as gv
from sklearn import tree
from sklearn.cross_validation import cross_val_score, train_test_split
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

	print ("РуZero value test: ", len(cols_for_model)-(len(header)-len(col_remove)))


	data = np.delete(data, col_remove,1)
	header = np.delete(header,col_remove,0)



	### Collecting data for model
	has_surv = (data[:,1] !='')
	y = data[has_surv,1]
	X = data[has_surv,2::]
	feature_names = header[2:]

	X_train, X_test, y_train, y_test = train_test_split(
				    X, y, test_size=0.5, random_state=111)

	print ('Survived train: ', sum(y_train=='1')/len(y_train))
	print ('Survived test: ', sum(y_test=='1')/len(y_test))

	### Best parameteres:
	###
	###
	#   On all dataset:
	#
	# 	min:  0.810810810811 [24, 110]
	#   mean:  0.842717473967 [14, 40]

	### Making prediction based on data sampled from train set
	min_val = 0
	param_min = [0,0]
	mean_val = 0
	param_mean = [0,0]
	test_val = 0
	param_test = [0,0]


	clf = ensemble.AdaBoostClassifier(
	    tree.DecisionTreeClassifier(min_samples_leaf=19),
	    n_estimators=10,
	    algorithm="SAMME", random_state=1567,
	    learning_rate=0.1)

	clf = clf.fit(X, y)
	scores = cross_val_score(clf, X, y, cv = 8, n_jobs=-1)


	print ("adaboost")
	print ("survived: ",sum (y=='1')/len(y))
	print (X.shape[0])
	print ("Crossvalidation scores: ")
	print (scores)
	print("min: ",scores.min(),"   mean:",scores.mean())


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