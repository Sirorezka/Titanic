

import csv as csv
import numpy as np
import graphviz as gv
from sklearn import tree
from sklearn.cross_validation import cross_val_score
from my_functions import *



def main ():

	my_project_dir = "kaggle_data\\"
	#print (my_project_dir + 'train.csv')


	#### READING TRAIN DATA ####

	csv_file_object = csv.reader(open(my_project_dir + 'train.csv', 'r')) 	# Load in the csv file
	header = next(csv_file_object) 						# Skip the fist line as it is a header
	data=[] 												# Create a variable to hold the data

	for row in csv_file_object: 							# Skip through each row in the csv file,
	    data.append(row[0:]) 								# adding each row to the data variable


	data_train = np.array(data) 							# Then convert from a list to an array.



	#### READING TEST DATA ####

	csv_file_object = csv.reader(open(my_project_dir + 'test.csv', 'r')) 	
	header1 = next(csv_file_object) 						
	data=[] 												

	for row in csv_file_object: 							
	    data.append(row[0:]) 								

	data_test = np.array(data) 							



	### JOINING TEST AND TRAIN DATA

	new_col = [""] * data_test.shape[0]
	new_col = np.array(new_col)
	data_test = np.column_stack((data_test[:,0], new_col,data_test[:,1:]))

	data_all = np.copy(np.row_stack ((data_train,data_test)))


	is_female = (data_all[0::,4]=='female')+0					# Converting sex to "is_female"
	data_all[0::,4] = is_female
	header[4] = "is_female"



	### NEW VARS  ###


	### Adding missing Fare data

	rows_with_fares =  (data_all[::,9]!='')
	all_fares = data_all[rows_with_fares,9].astype(np.float)
	fare_mean = all_fares.mean()
	data_all[data_all[::,9]=='',9] = fare_mean


	### PROCESSING AGE

	header = np.concatenate((header[0:6], ["Age_predict","Age_unknown","age_less_6","age_20_30"],header[6:]))
	new_col = np.array([""] * data_all.shape[0])
	data_all = np.column_stack((data_all[:,0:6],new_col,new_col,new_col,new_col,data_all[:,6:]))
	data_all = np.copy(data_all)

	age_not_known = (data_all[:,5] == "")     # if age is unknown
	#print (header[7])
	data_all[:,7] = 0
	data_all[age_not_known,7] = 1


	# if age is less than 6
	age_filter = list(map (lambda x: (x!="" and float(x)<6 and float(x)>=0), data_all[:,5]))
	age_filter = np.array(age_filter).astype(bool)
	data_all[:,8] = 0 
	data_all[age_filter,8] = 1


	# if age is more than 20
	age_filter = list(map (lambda x: (x!="" and float(x)<=30 and float(x)>=20), data_all[:,5]))
	age_filter = np.array(age_filter).astype(bool)
	data_all[:,9] = 0 
	data_all[age_filter,9] = 1


	# predicting age
	print (header)
	col_remove = [0, 1, 3, 5, 6,7,8,9,12,14,15]               # Column 5 represent age
	age_col = np.copy(data_all[:,5])
	data_age_model = np.copy (data_all)
	data_age_model = np.delete(data_age_model, col_remove,1)
	header_2 = np.copy(header)
	header_2 = np.delete(header_2,col_remove,0)
	
	print ("\n","age prediction model:")
	print(header_2,"\n")


	X = data_age_model
	feature_names = header_2

	has_age = (age_col!="")
	age_col_has_age = age_col[has_age].astype(float)
	X_has_age = np.copy(X[has_age,:])


	clf = tree.DecisionTreeRegressor(min_samples_leaf=50)
	clf = clf.fit(X_has_age,age_col_has_age)
	scores = cross_val_score(clf, X_has_age, age_col_has_age, cv = 5)
	print ("Crossvalidation scores for AGE prediction model: ")
	print (scores)
	print(scores.mean())
	print("\n")
	save_tree_img ("img/agetree.dot", clf, feature_names)

	X_no_age = np.copy(X[~has_age,:])
	age_predict = clf.predict(X_no_age)
	age_predict = np.array(age_predict).astype("float")
	age_predict = np.round(age_predict,2)


	# print (header[5:7])
	data_all[:,6] = np.copy(data_all[:,5])
	data_all[~has_age,6] = age_predict
	#print (data_all[:,5:8])


	# Processing Cabin type
	header = np.concatenate((header[0:15], ["Cabin_type","No_cabin"],header[15:]))
	new_col = np.array([""] * data_all.shape[0])
	data_all = np.column_stack((data_all[:,0:15],new_col,new_col,data_all[:,15:]))
	data_all = np.copy(data_all)

	def first_letter(x):
	    if x!="":
	        return x[0]
	    else:
	        return "N"
     
	cabin_type = list(map (first_letter,data_all[:,14]))      
	#print (header[15])
	data_all[:,15] = cabin_type
	data_all[:,16] = 0
	data_all[data_all[:,15]=="N",16] = 1


	# Processing Embarked

	for emb in np.unique(data_all[::,17]):
		header = np.append(header,"Embarked_%s" % str(emb))
		new_col = np.array([""] * data_all.shape[0])
		data_all = np.column_stack((data_all,new_col))
		data_all[:,len(header)-1] = 0
		data_all[data_all[::,17]==emb,len(header)-1] = 1
		#print (emb)

	print (header)


	# Count number of passengers in the same room
	print(header[12]) 
	header = np.append(header,"Same_Ticket")
	x1 = list(map (lambda x: sum(data_all[::,12]==x),data_all[::,12]))
	data_all = np.column_stack((data_all,x1))	
	print(data_all[::,(12,22)])



	# Count number survived in the same room
	print(header[12],header[3]) 
	header = np.append(header,"Same_Room_Surv")
	print(header[23]) 



	def same_room_surv (x_ticket,x_name, my_data):
	    #print (my_data[i_ind,8])
	    vect = np.column_stack((my_data[::,12]==x_ticket,my_data[:,1]=='1',my_data[:,3]!=x_name))
	    vect = np.all(vect,axis=1)
	    return(sum(vect))


	max_length = int(round(891*0.75,0))
	print (max_length)
	x1 = list(map (lambda x: same_room_surv(data_all[x,12],data_all[x,3],data_all[:max_length,]),range(data_all.shape[0])))
	data_all = np.column_stack((data_all,x1))	


	print (data_all.shape)

	predictions_file = open("kaggle_data/processed.csv", "w", newline='')
	predictions_file_object = csv.writer(predictions_file, delimiter=',',quotechar='"', quoting=csv.QUOTE_MINIMAL)
	predictions_file_object.writerow(header)	

	#np.savetxt(predictions_file_object, data_all, delimiter="\t")

	for i in range(data_all.shape[0]):														
	    predictions_file_object.writerow(data_all[i,:])		


if __name__ == "__main__":
    main()
