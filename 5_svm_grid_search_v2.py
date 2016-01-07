""" This simple code is desinged to teach a basic user to read in the files in python, simply find what proportion of males and females survived and make a predictive model based on this
Author : AstroDave
Date : 18 September 2012
Revised: 28 March 2014

"""
from multiprocessing import Pool

import csv as csv
import numpy as np
import graphviz as gv
from sklearn import tree, svm,preprocessing
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
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

	y = np.copy(data[has_surv,1].astype(int))
	X = np.copy(data[has_surv,2::].astype(float))
	feature_names = header[2:]




	#clf = svm.SVC(C=1, random_state=512)

	X_train, X_test, y_train, y_test = train_test_split(
				    X, y, test_size=0.75, random_state=333)

	step_gamma = (0.1-0.0001)/30 
	#print(np.arange(11, 17, step_gamma))
	#print (np.logspace(-9, 3, 25))
	step_C = (1000-1)/40 
	#print(np.arange(1, 1000, step_C))
	#print(np.logspace(-2, 4, 25))

	tuned_parameters = [{'C': np.logspace(-1, 3, 10) ,
						 'kernel': ['rbf'],
						 'gamma': np.logspace(-3, 1, 25)}]

	#quit()
	min_best_sc = 0
	mean_best_sc = 0
	test_best_sc = 0
	min_best_param = []
	mean_best_param = []
	test_best_param = []

	for i_C in tuned_parameters[0]['C']:
		for i_gamma in tuned_parameters[0]['gamma']:
			print (i_C,i_gamma)

			clf = svm.SVC(C=i_C,random_state=512,gamma=i_gamma)
			scores = cross_val_score(clf, X_train, y_train, cv = 8, n_jobs=-1)
			
			clf.fit(X_train, y_train)

			y_true, y_pred = y_test, clf.predict(X_test)
			score_surv = sum(y_true[y_true==1]==y_pred[y_true==1])/len(y_true[y_true==1])
			score_surv2 = sum(y_true[y_true==0]==y_pred[y_true==0])/len(y_true[y_true==0])
			score_surv3 = sum(y_true==y_pred)/len(y_true)

			#print ("sc 1: ", score_surv,"sc 0: ",score_surv2,"sc all:", score_surv3)
			print ("sc min: ",scores.min(),"sc mean: ",scores.mean(),"sc test: ",score_surv3)
			#print(classification_report(y_true, y_pred, digits=4))
			print()

			if scores.min()>min_best_sc:
				min_best_sc = scores.min()
				min_best_param= [i_C,i_gamma]

			if scores.mean()>mean_best_sc:
				mean_best_sc = scores.mean()
				mean_best_param = [i_C,i_gamma]

			if score_surv3>test_best_sc:
				test_best_sc = score_surv3
				test_best_param = [i_C,i_gamma]

	print ("Min: ",min_best_sc,"   ",min_best_param )
	print ("Mean: ",mean_best_sc,"   ",mean_best_param )
	print ("Min test: ",test_best_sc,"   ",test_best_param )


	clf = svm.SVC(C=100,random_state=512,gamma=0.005)
	clf.fit(X, y)	
	#print(clf.best_params_)
	X_nolabel = data[~has_surv,2::]
	x_id = data[~has_surv,0]
	y_predict = clf.predict(X_nolabel).astype(int)

	print ("\ntest predict")
	print ("survived: ",sum (y_predisct==1)/len(y_predict))


	predictions_file = open("output/svm.csv", "w", newline='')
	predictions_file_object = csv.writer(predictions_file)
	predictions_file_object.writerow(["PassengerId", "Survived"])	
	#predictions_file_object.writerow([pass_id, y_predict])

	for i in range(len(y_predict)):														
	    predictions_file_object.writerow([x_id[i], y_predict[i]])
		
														
if __name__ == "__main__":
    main()