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


	print (header)
	print(data[35:44,:])

	print(data[1:15,:])



	y = data[0::,0]
	X = data[0::,1::]
	feature_names = header[1:]


	rand_range = range(0, X.shape[0])
	np.random.choice(rand_range, size=5, replace=False, p=None)


	print (len(header))
	print (X)
	clf = tree.DecisionTreeClassifier(min_samples_leaf=20)
	clf = clf.fit(X,y)

	scores = cross_val_score(clf, X, y, cv = 8)
	print (scores)
	print(scores.mean())


	save_tree_img ("img/tree.dot", clf, feature_names, class_names =["dead","survived"])


	quit()
	# Read test data
	test_file = open(my_project_dir + 'test.csv', 'r')
	test_file_object = csv.reader(test_file)
	header = next(test_file_object)

	# Also open the a new file so I can write to it. Call it something descriptive
	# Finally, loop through each row in the train file, and look in column index [3] (which is 'Sex')
	# Write out the PassengerId, and my prediction.

	predictions_file = open("gendermodel.csv", "w")
	predictions_file_object = csv.writer(predictions_file)
	predictions_file_object.writerow(["PassengerId", "Survived"])	# write the column headers
	for row in test_file_object:									# For each row in test file,
	    if row[3] == 'female':										# is it a female, if yes then
	        predictions_file_object.writerow([row[0], "1"])			# write the PassengerId, and predict 1
	    else:														# or else if male,
	        predictions_file_object.writerow([row[0], "0"])			# write the PassengerId, and predict 0.

	        
	test_file.close()												# Close out the files.
	predictions_file.close()





if __name__ == "__main__":
    main()