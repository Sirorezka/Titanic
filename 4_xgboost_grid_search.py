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
	X_nolabel = data[~has_surv,2::]
	x_id = data[~has_surv,0]
	feature_names = header[2:]


	X_train, X_test, y_train, y_test = train_test_split(
				    X, y, test_size=0.5, random_state=222)

	# specify validations set to watch performance
	print (header)
	T_train_xgb = xgb.DMatrix(X_train.astype(float), label=y_train.astype(float))
	T_test_xgb = xgb.DMatrix(X_test.astype(float))



	## Error function
	def evalerror(preds, dtrain):
	    labels = dtrain.get_label()
	    y_predict = list(map (lambda x: int(x>0.5), preds))
	    return 'myerror', sum(labels != y_predict) / len(labels)


	file_path = "output/xgboost_params.csv"
	xgb_params_file = open(file_path, "w", newline='')
	xgb_params_file_object = csv.writer(xgb_params_file)
	xgb_params_file.close()


	## n_round = 4
	## max_depth = 6
	## i_ets = 0.7
	## i_subsample = 0.6
	## i_lambda = 0.9

	for i_num_round in range(6,7,1):
		for i_max_depth in range(8,9,1):
			for i_eta in range(7,8,1):
				for i_subsample in range(6,7,1):
					for i_lambda in range(9,10,1):

						num_round = i_num_round
						param = {'max_depth':i_max_depth, 'eta':i_eta/10.0, 
								 'silent':1, 'subsample':i_subsample/10.0,
								 'lambda':i_lambda/10.0, 'num_round': 4,
								 'objective':'binary:logistic'}
					
						#param['nthread'] = 4
						param['eval_metric'] = 'logloss'

						d_evals_result = {}
						eval_hist = xgb.cv(param, T_train_xgb, num_round, nfold=8,
					      					 metrics={'logloss'}, seed = 25, 
					      					 show_progress =False, show_stdv =True,
					      					 feval=evalerror)
						#print (eval_hist)
						scores_cv = 1-  eval_hist['test-myerror-mean'][num_round-1]
						# print ("Cross validation score: ", scores_cv)

						bst = xgb.train(param, T_train_xgb, num_round)
						y_predict = bst.predict(T_test_xgb)
						y_predict = list(map (lambda x: int(x>0.6), y_predict))
						y_predict = np.array(y_predict).astype(float)
						score_test = sum(y_test==y_predict)/len(y_test)

						#print (i_num_round,i_max_depth)
						print (i_num_round,i_max_depth," -- ",scores_cv,score_test)

						with open(file_path, 'a', newline='') as f:
							f_object = csv.writer(f)
							f_object.writerow([i_num_round, 
											   i_max_depth, round(i_eta/10.0,2),
											   round(i_subsample/10.0,2), round(i_lambda/10.0,2),
											   scores_cv, score_test
												])




	T_train_xgb = xgb.DMatrix(X.astype(float), label=y.astype(float))
	X = data[:,2::]
	x_id = data[:,0]
	T_test_xgb = xgb.DMatrix(X.astype(float))

	bst = xgb.train(param, T_train_xgb, num_round)	

	y_predict = bst.predict(T_test_xgb)
	y_probab = np.copy(y_predict)
	#print (y_predict)
	y_predict = list(map (lambda x: int(x>0.6), y_predict))
	y_predict = np.array(y_predict).astype(int)

	print (sum(y_predict==1)/len(y_predict))

	predictions_file = open("output/prob_xgboost.csv", "w", newline='')
	predictions_file_object = csv.writer(predictions_file)
	predictions_file_object.writerow(["PassengerId", "Survived"])	
	#predictions_file_object.writerow([pass_id, y_predict])

	for i in range(len(y_predict)):	
		#predictions_file_object.writerow([x_id[i], y_predict[i]])														
	    predictions_file_object.writerow([x_id[i], y_predict[i],1-y_probab[i],y_probab[i]])		

if __name__ == "__main__":
	main()