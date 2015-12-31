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
	my_project_dir = "kaggle_data\\"
	print (my_project_dir + 'train.csv')

	csv_file_object = csv.reader(open(my_project_dir + 'train.csv', 'r')) 	# Load in the csv file
	header = next(csv_file_object) 						# Skip the fist line as it is a header
	data=[] 												# Create a variable to hold the data

	for row in csv_file_object: 							# Skip through each row in the csv file,
	    data.append(row[0:]) 								# adding each row to the data variable


	print (type(data))
	data = np.array(data) 									# Then convert from a list to an array.


	print (data.shape)										# Get array shape
	print (header)
	print (data[1,])
	print (len(data))



	is_female = (data[0::,4]=='female')+0					# Converting sex to "is_female"
	data[0::,4] = is_female
	header[4] = "is_female"


	col_remove = [0, 3, 5, 8, 10,11]               # Column 5 represent age
	data = np.delete(data, col_remove,1)
	header = np.delete(header,col_remove,0)



	### Collecting data for model
	y = data[0::,0]
	X = data[0::,1::]
	feature_names = header[1:]


	### Making prediction based on data sampled from train set

	X_train, y_train, X_test, y_test = generate_data_train_test (X,y,0.75)

	clf = tree.DecisionTreeClassifier(min_samples_leaf=20)
	clf = clf.fit (X_train,y_train)
	y_predict = clf.predict(X_test)

	predict_rate = round(sum(y_predict==y_test)/len(y_predict),2)
	print ("\nPrediction score: ",predict_rate,"\n")




	clf = tree.DecisionTreeClassifier(min_samples_leaf=20)
	clf = clf.fit(X,y)
	scores = cross_val_score(clf, X, y, cv = 8)
	print ("Crossvalidation scores: ")
	print (scores)
	print(scores.mean())
	print ("\n")


	save_tree_img ("img/tree.dot", clf, feature_names, class_names =["dead","survived"])



	# Read test data
	test_file = open(my_project_dir + 'test.csv', 'r')
	test_file_object = csv.reader(test_file)
	header_train = next(test_file_object)

	data_train = []
	for row in test_file_object: 							# Skip through each row in the csv file,
	    data_train.append(row[0:]) 

	data_train = np.array(data_train)

	is_female = (data_train[0::,3]=='female')+0					# Converting sex to "is_female"
	data_train[0::,3] = is_female
	header_train[3] = "is_female"


	print (header_train)
	col_remove = [0, 2, 4, 7, 9,10]               # Column 5 represent age
	pass_id = data_train[:,0]
	data_train = np.delete(data_train, col_remove,1)
	header_train = np.delete(header_train, col_remove,0)

	# predicting missing fare:
	rows_with_fares =  (data_train[::,4]!='')
	all_fares = data_train[rows_with_fares,4].astype(np.float)
	fare_mean = all_fares.mean()
	data_train[data_train[::,4]=='',4] = fare_mean

	#fare_predict = all_fares.sum()/all

	y_predict = clf.predict(data_train)
	pass_id, y_predict



	predictions_file = open("output/decisionforest.csv", "w")
	predictions_file_object = csv.writer(predictions_file)
	predictions_file_object.writerow(["PassengerId", "Survived"])	
	#predictions_file_object.writerow([pass_id, y_predict])

	for i in range(y_predict.shape[0]):														
	    predictions_file_object.writerow([pass_id[i], y_predict[i]])			
														

	        
	test_file.close()												
	predictions_file.close()





if __name__ == "__main__":
    main()