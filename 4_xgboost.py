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
from sklearn.metrics import confusion_matrix
from sklearn import ensemble 
from sklearn import linear_model
from my_functions import *
from sklearn import svm
import xgboost as xgb

def main ():

	## LOADING DATA

	my_project_dir = "kaggle_data\\"
	csv_file_object = csv.reader(open(my_project_dir + 'processed.csv', 'r')) 	# Load in the csv file
	header = next(csv_file_object) 						# Skip the fist line as it is a header
	data=[] 												# Create a variable to hold the data

	for row in csv_file_object: 							# Skip through each row in the csv file,
	    data.append(row[0:]) 								# adding each row to the data variable


	data = np.array(data) 									# Then convert from a list to an array.


	###  Select columns for model:
	
	cols_for_model = ["PassengerId", "Survived","Pclass","Mrs","Mr","Miss","Master",
						"Captain","Other","is_female","Age_predict","Age_unknown",
						"age_less_10","age_10_20","age_20_30","age_30_50","age_more_50",
						"SibSp","Parch","Fare","No_cabin","Embarked_C","Embarked_Q",
						"Embarked_S","Same_Ticket"
						,"Same_Room_Surv","Same_Room_surv_perc"
						]

	###  Removing columns that we don't need
	col_remove = []
	for i in range(len(header)):
		if (header[i] not in cols_for_model): col_remove.append(i)

	print ("Zero value test: ", len(cols_for_model)-(len(header)-len(col_remove)))
	data = np.delete(data, col_remove,1)
	header = np.delete(header,col_remove,0)


	###   Collecting data for model
	has_surv = (data[:,1] !='')
	y = data[has_surv,1].astype(float)
	X = data[has_surv,2::]
	feature_names = header[2:]


	X_train, X_test, y_train, y_test = train_test_split(
				    X, y, test_size=0.5, random_state=222)

	# specify validations set to watch performance
	print (header)
	T_train_xgb = xgb.DMatrix(X_train.astype(float), label=y_train.astype(float))
	T_test_xgb = xgb.DMatrix(X_test.astype(float))



	def evalerror(preds, dtrain):
	    labels = dtrain.get_label()
	    y_predict = list(map (lambda x: int(x>0.5), preds))
	    return 'myerror', sum(labels != y_predict) / len(labels)

	num_round = 7
	param = {'max_depth':5, 'eta':0.7, 
			 'silent':1, 'subsample':0.5,
			 'lambda':0.1,
			 'objective':'binary:logistic'}


	#param['nthread'] = 4
	param['eval_metric'] = 'logloss'

	d_evals_result = {}
	eval_hist = xgb.cv(param, T_train_xgb, num_round, nfold=8,
      					 metrics={'logloss'}, seed = 25, 
      					 show_progress =True, show_stdv =True,
      					 feval=evalerror)
	print (eval_hist)
	scores_cv = eval_hist['test-myerror-mean'][num_round-1]
	print ("Cross validation score: ", scores_cv)

	bst = xgb.train(param, T_train_xgb, num_round)
	# xgb.plot_importance(bst)	
	# bst.dump_model('img/dump.raw.txt','img/featmap.txt')

	y_predict = bst.predict(T_test_xgb, output_margin=True)
	y_predict = list(map (lambda x: int(x>0.5), y_predict))
	y_predict = np.array(y_predict).astype(float)
	#print (y_predict)

	print ("\nConfustion Matrix:")
	print (confusion_matrix(y_test,y_predict))
	print ("Percent matches score: ",sum(y_test==y_predict)/len(y_test))
	print ("\n")
	score_test = sum(y_test==y_predict)/len(y_test)

	y_predict = bst.predict(T_test_xgb, output_margin=True)
	y_predict = list(map (lambda x: int(x>0.5), y_predict))
	y_predict = np.array(y_predict).astype(int)
	#print (y_predict)

	predictions_file = open("output/xgboost.csv", "w", newline='')
	predictions_file_object = csv.writer(predictions_file)
	predictions_file_object.writerow(["PassengerId", "Survived"])	
	#predictions_file_object.writerow([pass_id, y_predict])

	for i in range(len(y_predict)):														
	    predictions_file_object.writerow([x_id[i], y_predict[i]])		

if __name__ == "__main__":
    main()