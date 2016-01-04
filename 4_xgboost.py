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


	# for i in range(len(header)):
	# 	print (i," ",header[i])


	col_remove = [3, 5, 8, 12,14,15,17,18]               # Column 5 represent age
	data = np.delete(data, col_remove,1)
	header = np.delete(header,col_remove,0)



	### Collecting data for model
	has_surv = (data[:,1] !='')
	#print (has_surv)
	n_par = int(sum(has_surv)*0.75) 
	
	# y = data[0:n_par,1]
	# X = data[0:n_par,2::]
	y = data[has_surv,1].astype(float)
	X = data[has_surv,2::]
	x_id = data[~has_surv,0]
	X_test = data[~has_surv,2::]
	feature_names = header[2:]



	# specify validations set to watch performance
	print (header)
	T_train_xgb = xgb.DMatrix(X.astype(float), label=y.astype(float))
	T_test_xgb = xgb.DMatrix(X_test.astype(float))



	def evalerror(preds, dtrain):
	    labels = dtrain.get_label()
	    y_predict = list(map (lambda x: int(x>0.5), preds))
	    return 'myerror', sum(labels != y_predict) / len(labels)

	num_round = 10
	param = {'max_depth':6, 'eta':0.7, 'silent':1, 'objective':'binary:logistic'}
	#param['nthread'] = 4
	param['eval_metric'] = 'logloss'

	d_evals_result = {}
	eval_hist = xgb.cv(param, T_train_xgb, num_round, nfold=8,
      					 metrics={'logloss'}, seed = 0, 
      					 show_progress =True, show_stdv =True,
      					 feval=evalerror)
	print (eval_hist)

	bst = xgb.train(param, T_train_xgb, num_round)
	xgb.plot_importance(bst)	
	bst.dump_model('img/dump.raw.txt','img/featmap.txt')

	y_predict = bst.predict(T_train_xgb, output_margin=True)
	y_predict = list(map (lambda x: int(x>0.5), y_predict))
	y_predict = np.array(y_predict).astype(float)
	#print (y_predict)
	print ("\nConfustion Matrix:")
	print (confusion_matrix(y,y_predict))
	print ("\n")

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