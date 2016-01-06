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
from my_functions import *



def main ():

	## LOADING DATA

	my_project_dir = "kaggle_data\\"
	csv_file_object = csv.reader(open(my_project_dir + 'processed.csv', 'r')) 	# Load in the csv file
	header = next(csv_file_object) 						# Skip the fist line as it is a header
	data=[] 												# Create a variable to hold the data

	for row in csv_file_object: 							# Skip through each row in the csv file,
	    data.append(row[0:]) 								# adding each row to the data variable


	# print (type(data))
	data = np.array(data) 									# Then convert from a list to an array.


	print (header)
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
	print (has_surv)

	y = data[has_surv,1]
	X = data[has_surv,2::]
	feature_names = header[2:]


	### Making prediction based on data sampled from train set

	min_score = 0
	mean_score = 0
	min_i = 0
	mean_i = 0

	
	### Best params:
	### optimal params is min_samples_leaf =  12

	# for i in range(1,100):
	clf = tree.DecisionTreeClassifier(min_samples_leaf=12)
	clf = clf.fit(X,y)
	scores = cross_val_score(clf, X, y, cv = 8)
	print (X.shape[0])
	print ("Crossvalidation scores: ")
	print (scores)
	print("min: ",scores.min(),"  mean: ", scores.mean())
	print ("\n")
		# if (scores.min()>min_score):
		# 		min_i = i
		# 		min_score = scores.min()
		# if (scores.mean()>mean_score):
		# 		mean_i = i
		# 		mean_score = scores.mean()

	print ("data:")
	print (min_i,min_score)
	print (mean_i,mean_score)

	save_tree_img ("img/tree.dot", clf, feature_names, class_names =["dead","survived"])


	# predicting data
	x_id = data[~has_surv,0]
	X = data[~has_surv,2::]

	# x_id = data[:,0]
	# X = data[:,2::]
	y_predict = clf.predict(X)
	#print (y_predict)


	predictions_file = open("output/decisiontree.csv", "w", newline='')
	predictions_file_object = csv.writer(predictions_file)
	predictions_file_object.writerow(["PassengerId", "Survived"])	
	#predictions_file_object.writerow([pass_id, y_predict])

	for i in range(len(y_predict)):														
	    predictions_file_object.writerow([x_id[i], y_predict[i]])			
														
if __name__ == "__main__":
    main()