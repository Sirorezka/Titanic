""" This simple code is desinged to teach a basic user to read in the files in python, simply find what proportion of males and females survived and make a predictive model based on this
Author : AstroDave
Date : 18 September 2012
Revised: 28 March 2014

"""


import csv as csv
import numpy as np
from sklearn import tree




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

	print ("\n")
	print ("before removing")
	print(header)
	print (data[0:,0::])

	print ("\n")
	print ("after removing")
	data = np.delete(data,[3,8,11],1)
	print (header)
	print(data[0::,0::])


	clf = tree.DecisionTreeClassifier()
	clf = clf.fit( data[0::,2::],data[0::,1])

	quit()





	# First, read in test.csv
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