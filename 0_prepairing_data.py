

import re
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



	p_match = re.compile("\w*\.")
	arg_match = list(map(lambda x: p_match.findall(x)[0],data_all[:,3]))

	### adding title
	header = np.concatenate((header[0:4], ["Title","Mrs", "Mr", "Miss", "Master","Captain","Other"],header[4:]))
	new_col = np.array([""] * data_all.shape[0])
	data_all = np.column_stack((data_all[:,0:4],new_col,new_col,new_col,new_col,new_col,new_col,new_col,data_all[:,4:]))
	data_all = np.copy(data_all)
	
	data_all [:,4] = arg_match
	arg_match = np.array(arg_match)
	data_all [arg_match=="Ms.",4] = "Mrs."
	data_all [arg_match=="Mlle.",4] = "Miss."
	data_all [arg_match=="Mme.",4] = "Miss."

	# processing most common titles
	data_all [:,5] = 0 
	data_all [data_all [:,4]=="Mrs.",5] = 1
	data_all [:,6] = 0 
	data_all [data_all [:,4]=="Mr.",6] = 1
	data_all [:,7] = 0 
	data_all [data_all [:,4]=="Miss.",7] = 1
	data_all [:,8] = 0 
	data_all [data_all [:,4]=="Master.",8] = 1
	data_all [:,9] = 0 
	data_all [data_all [:,4]=="Capt.",9] = 1

	# 'Other' title
	is_title = np.sum(data_all[::,5:10].astype(int),dtype=int,axis=1)
	data_all [:,10] = 0 
	data_all [is_title==0,10] = 1



	### PROCESSING AGE

	
	header = np.concatenate((header[0:13], ["Age_predict","Age_unknown","age_less_10","age_10_20","age_20_30","age_30_50","age_more_50"],header[13:]))
	new_col = np.array([""] * data_all.shape[0])
	data_all = np.column_stack((data_all[:,0:13],new_col,new_col,new_col,new_col,new_col,new_col,new_col,data_all[:,13:]))
	data_all = np.copy(data_all)



	age_not_known = (data_all[:,12] == "")     # if age is unknown
	#print (header[7])
	data_all[:,14] = 0
	data_all[age_not_known,14] = 1



	# PREDICTING MISSING AGE
	col_remove = []
	for i in range(len(header)):
		if (header[i] not in ["Pclass","Mrs","Mr","Miss","Master","Other","is_female",
				"SibSp","Parch","Fare"]): col_remove.append(i)

	#print ([1:11])
	#print (len(col_remove),": ",col_remove)

	age_col = np.copy(data_all[:,12])
	data_age_model = np.copy (data_all)
	data_age_model = np.delete(data_age_model, col_remove,1)
	header_2 = np.copy(header)
	header_2 = np.delete(header_2,col_remove,0)


	print ("\n","~~~  age prediction model  ~~~")
	print ("used parameters")
	print(header_2,"\n")

	X = data_age_model
	feature_names = header_2

	has_age = (age_col!="")
	age_col_has_age = age_col[has_age].astype(float)
	X_has_age = np.copy(X[has_age,:])


	clf = tree.DecisionTreeRegressor(min_samples_leaf=40)
	clf = clf.fit(X_has_age,age_col_has_age)
	scores = cross_val_score(clf, X_has_age, age_col_has_age, cv = 5)
	print ("Crossvalidation scores for AGE prediction model: ")
	print (scores)
	print("min score:",scores.min()," ||  mean score:",scores.mean())
	print("\n")
	save_tree_img ("img/agetree.dot", clf, feature_names)

	X_no_age = np.copy(X[~has_age,:])
	age_predict = clf.predict(X_no_age)
	age_predict = np.array(age_predict).astype("float")
	age_predict = np.round(age_predict,2)



	data_all[:,13] = np.copy(data_all[:,12])
	data_all[~has_age,13] = age_predict
	#print (data_all[:,5:8])

	## GROUPING AGES

	age_predict = data_all[:,13]
	# if age is less than 10
	age_filter = list(map (lambda x: (x!="" and float(x)<=10 and float(x)>=0), age_predict))
	age_filter = np.array(age_filter).astype(bool)
	data_all[:,15] = 0 
	data_all[age_filter,15] = 1


	# if age is more than 10 less than 20
	age_filter = list(map (lambda x: (x!="" and float(x)<=20 and float(x)>10), age_predict))
	age_filter = np.array(age_filter).astype(bool)
	data_all[:,16] = 0 
	data_all[age_filter,16] = 1

	# if age is more than 20 less than 30
	age_filter = list(map (lambda x: (x!="" and float(x)<=30 and float(x)>20), age_predict))
	age_filter = np.array(age_filter).astype(bool)
	data_all[:,17] = 0 
	data_all[age_filter,17] = 1


	# if age is more than 30 less than 50
	age_filter = list(map (lambda x: (x!="" and float(x)<=50 and float(x)>30), age_predict))
	age_filter = np.array(age_filter).astype(bool)
	data_all[:,18] = 0 
	data_all[age_filter,18] = 1


	# if age is more than 50
	age_filter = list(map (lambda x: (x!="" and float(x)>50), age_predict))
	age_filter = np.array(age_filter).astype(bool)
	data_all[:,19] = 0 
	data_all[age_filter,19] = 1

	#print (header[14:20])
	#print(np.sum(data_all[:,14:20].astype(float),axis=0))
	


	## CABIN TYPE

	# Processing Cabin type
	header = np.concatenate((header[0:25], ["Cabin_type","No_cabin"],header[25:]))
	new_col = np.array([""] * data_all.shape[0])
	data_all = np.column_stack((data_all[:,0:25],new_col,new_col,data_all[:,25:]))
	data_all = np.copy(data_all)


	def first_letter(x):
	    if x!="":
	        return x[0]
	    else:
	        return "N"
     
	cabin_type = list(map (first_letter,data_all[:,24]))      
	#print (header[15])
	data_all[:,25] = cabin_type
	data_all[:,26] = 0
	data_all[data_all[:,25]=="N",26] = 1

	#print (header[24:27])
	#print (data_all[:,24:27])
	


	### PROCESSING EMBARKED


	for emb in np.unique(data_all[::,27]):
		header = np.append(header,"Embarked_%s" % str(emb))
		new_col = np.array([""] * data_all.shape[0])
		data_all = np.column_stack((data_all,new_col))
		data_all[:,len(header)-1] = 0
		data_all[data_all[::,27]==emb,len(header)-1] = 1
		#print (emb)

	###  print (header[27:32])
	###  print (data_all[:,27:32])


	### TWO ADD. COMPUTED VARS

	# Count number of passengers in the same room
	#print(header[22]) 
	header = np.append(header,"Same_Ticket")
	x1 = list(map (lambda x: sum(data_all[::,22]==x),data_all[::,22]))
	data_all = np.column_stack((data_all,x1))	
	#print(data_all[::,(22,32)])



	# Count number survived in the same room
	print(header[22],header[3]) 
	header = np.append(header,["Same_Room_Surv","Same_Room_surv_perc"])
	print(header[33]) 


	def same_room_surv (x_ticket,x_name, my_data):
	    #print (my_data[i_ind,8])
	    vect = np.column_stack((my_data[::,22]==x_ticket,my_data[:,1]=='1',my_data[:,3]!=x_name))
	    vect = np.all(vect,axis=1)
	    return(sum(vect))

	def same_room_surv_perc (x_ticket,x_name, my_data):
	    #print (my_data[i_ind,8])
	    vect = np.column_stack((my_data[::,22]==x_ticket,my_data[:,1]=='1',my_data[:,3]!=x_name))
	    vect_neigh = np.column_stack((my_data[::,22]==x_ticket,my_data[:,3]!=x_name))
	    vect = np.all(vect,axis=1)
	    vect_neigh = np.all(vect_neigh,axis=1)
	    perct = 0

	    if (sum(vect_neigh)!=0):
	    	perct = round(sum(vect)/sum(vect_neigh),2)
	    return(perct)

	max_length = int(round(891*0.75,0))  ## sample length for computing var is 75% of test data
	print (max_length)
	x1 = list(map (lambda x: same_room_surv(data_all[x,22],data_all[x,3],data_all[:max_length,]),range(data_all.shape[0])))
	x2 = list(map (lambda x: same_room_surv_perc(data_all[x,22],data_all[x,3],data_all[:max_length,]),range(data_all.shape[0])))

	data_all = np.column_stack((data_all,x1,x2))	


	print (data_all.shape)
	#print (header)

	predictions_file = open("kaggle_data/processed.csv", "w", newline='')
	predictions_file_object = csv.writer(predictions_file, delimiter=',',quotechar='"', quoting=csv.QUOTE_MINIMAL)
	predictions_file_object.writerow(header)	

	#np.savetxt(predictions_file_object, data_all, delimiter="\t")

	for i in range(data_all.shape[0]):														
	    predictions_file_object.writerow(data_all[i,:])		


if __name__ == "__main__":
    main()
