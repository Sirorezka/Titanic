""" This simple code is desinged to teach a basic user to read in the files in python, simply find what proportion of males and females survived and make a predictive model based on this
Author : AstroDave
Date : 18 September 2012
Revised: 28 March 2014

"""


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
				    X, y, test_size=0.5, random_state=111)

	tuned_parameters = [{'C': [1, 5, 10, 15, 20, 100, 1000],
						 'kernel': ['rbf'],
						 'gamma': [0.001, 0.0001, 0.005, 0.01, 0.0005]}]



	# for i_C in tuned_parameters[0]['C']:
	# 	for i_gamma in tuned_parameters[0]['gamma']:
	# 		print (i_C,i_gamma)

	scores = ['precision','precision_weighted','average_precision']

	for score in scores:
	    print("# Tuning hyper-parameters for %s" % score)
	    print()

	    clf = GridSearchCV(svm.SVC(C=1,random_state=512), tuned_parameters, cv=5,
	                       scoring='%s' % score,
	                       n_jobs  = -1,verbose =1)
	    clf.fit(X_train, y_train)

	    print("Best parameters set found on development set:")
	    print()
	    print(clf.best_params_)
	    print()
	    print("Grid scores on development set:")
	    print()

	    mean_scores = list(map(lambda x: x[1],clf.grid_scores_))
	    mean_scores = np.array(mean_scores)
	    indices = np.argsort(mean_scores)


	    for params, mean_score, scores in np.array(clf.grid_scores_)[indices]:
	        print("%0.3f (+/-%0.03f) for %r"
	              % (mean_score, scores.std() * 2, params))

	        #clf = svm.SVC(params)
	        #clf.fit(X_train, y_train)

	    print()

	    print("Detailed classification report:")
	    print()
	    print("The model is trained on the full development set.")
	    print("The scores are computed on the full evaluation set.")
	    print()
	    y_true, y_pred = y_test, clf.predict(X_test)
	    print(classification_report(y_true, y_pred, digits=4))
	    print()


	clf = svm.SVC(C=15,random_state=512,gamma=0.001,probability=True)
	clf.fit(X, y)	
	#print(clf.best_params_)
	X_nolabel = data[:,2::]
	x_id = data[:,0]
	y_predict = clf.predict(X_nolabel).astype(int)
	y_prob = clf.predict_proba(X_nolabel)

	#print (y_prob)
	print ("\ntest predict")
	print ("survived: ",sum (y_predict==1)/len(y_predict))


	predictions_file = open("output/prob_svm.csv", "w", newline='')
	predictions_file_object = csv.writer(predictions_file)
	predictions_file_object.writerow(["PassengerId", "Survived"])	
	#predictions_file_object.writerow([pass_id, y_predict])

	for i in range(len(y_predict)):														
	    predictions_file_object.writerow([x_id[i], y_predict[i],y_prob[i,0],y_prob[i,1]])
		
														
if __name__ == "__main__":
    main()