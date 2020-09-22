import pandas as pd
import numpy as np
import pprint
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score

import seaborn as sns
import matplotlib.pyplot as plt

#EX1 & EX2

#Read result, train and test files as dataframe
test_df = pd.read_csv(Path(Path.cwd() / 'data' / 'test.tsv'), sep='\t', names=['Date', 'Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio'])
train_df = pd.read_csv(Path(Path.cwd() / 'data' / 'train.tsv'), sep='\t', names=['Occupancy', 'Date', 'Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio'])
results_df = pd.read_csv(Path(Path.cwd() / 'data' / 'results.tsv'), sep='\t', names=['Occupancy'])

#Remove na fields
test_df.dropna(inplace = True)
train_df.dropna(inplace = True)
results_df.dropna(inplace = True)

#Join results data with test_df
test_df = pd.merge(test_df, results_df, how='inner', left_index=True, right_index=True)

#Prepare test/train datasets
dataset = {'test': {'y':test_df['Occupancy'],'x_single':test_df[['Temperature']],'x_multi':test_df[['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']]},
			'train': {'y':train_df['Occupancy'],'x_single':train_df[['Temperature']],'x_multi':train_df[['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']]}}

#Train one and multivariable model
dataset['singlevar_model'] = LogisticRegression().fit(dataset['train']['x_single'], dataset['train']['y'])
dataset['multivar_model'] = LogisticRegression().fit(dataset['train']['x_multi'], dataset['train']['y'])

for data_type in ['train', 'test']:
	#Make predictions for both models using test/train data and calculate acc, sens, specificity, f1 score and beta scores for them
	dataset[data_type]['y_single_predictions'] = dataset['singlevar_model'].predict(dataset[data_type]['x_single'])
	dataset[data_type]['y_multi_predictions'] = dataset['multivar_model'].predict(dataset[data_type]['x_multi'])

	for var in ['single','multi']:
		tn, fp, fn, tp = confusion_matrix(dataset[data_type]['y'], dataset[data_type]['y_'+ var +'_predictions']).ravel()
		dataset[data_type]['y_' + var +'_parameters'] = {'accuracy' : accuracy_score(dataset[data_type]['y'], dataset[data_type]['y_'+ var +'_predictions']),
													   'sensitivity' : recall_score(dataset[data_type]['y'], dataset[data_type]['y_'+ var +'_predictions']),
													   'speciality' : tn / (tn + fp)}
		#Calculate F1 score
		dataset[data_type]['f_1_'+var] = f1_score(dataset[data_type]['y'], dataset[data_type]['y_'+ var +'_predictions'])

		#Someone should be in a room
		dataset[data_type]['f_beta_first_case_'+var] = fbeta_score(dataset[data_type]['y'], dataset[data_type]['y_'+ var +'_predictions'], beta=2)

		#Room is empty
		dataset[data_type]['f_beta_second_case_'+var] = fbeta_score(dataset[data_type]['y'], dataset[data_type]['y_'+ var +'_predictions'], beta=0.5)

#Save predicted data from both models to out file
output_df = pd.DataFrame({'single': dataset['test']['y_single_predictions'], 'multi': dataset['test']['y_multi_predictions']})
output_df.to_csv(Path(Path.cwd() / 'data' / 'out.tsv'), index=False, header=False)

for data_type in ['test','train']:
	print(data_type)
	for var in ['single','multi']:
		print("Params_" + var + ": ", dataset[data_type]['y_' + var +'_parameters'])
		print("F1_" + var + ": ", dataset[data_type]['f_1_'+var])
		print("F_Beta_V1_" + var + ": ", dataset[data_type]['f_beta_first_case_'+var])
		print("F_Beta_V2_" + var + ": ", dataset[data_type]['f_beta_second_case_'+var])