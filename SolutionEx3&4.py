import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

#EX3

#Read data
data_df = pd.read_csv(Path(Path.cwd() / "survey_results_public.csv"), usecols=['Respondent', 'Age', 'YearsCodePro', 'ConvertedComp', 'Hobbyist', 'Student', 'Dependents'], index_col='Respondent')

#Replace strings inside YearsCode & YerasCodePro & Student column
data_df.replace(to_replace={'Less than 1 year':'0', 'More than 50 years':'50', 'Younger than 5 years':'5', 'Older than 85':'85','Yes': '1','No': '0','Yes, part-time': '1','Yes, full-time': '1'}, inplace=True)

#Drop empty records & convert values to int
data_df.dropna(inplace=True, how='any')
data_df = data_df.astype('int64')

#Prepare dataset
dataset = {'y':data_df.Hobbyist,'x':data_df[['Age', 'YearsCodePro', 'ConvertedComp', 'Student', 'Dependents']]}

#Trainmodel
dataset['model'] = LogisticRegression().fit(dataset['x'], dataset['y'])

#Make predictions and calculate acc, sens, specificity, f1 score and beta scores for them
dataset['y_predictions'] = dataset['model'].predict(dataset['x'])

tn, fp, fn, tp = confusion_matrix(dataset['y'], dataset['y_predictions']).ravel()
dataset['y_parameters'] = {'accuracy' : accuracy_score(dataset['y'], dataset['y_predictions']),
							'sensitivity' : recall_score(dataset['y'], dataset['y_predictions']),
							'speciality' : tn / (tn + fp)}
#Calculate F1 score
dataset['f_1_'] = f1_score(dataset['y'], dataset['y_predictions'])

#EX4
train_x, test_x, train_y, test_y = train_test_split(data_df[['Age', 'YearsCodePro', 'ConvertedComp', 'Student', 'Dependents']], data_df.Hobbyist, test_size=0.25, random_state=5)

#Train data
log_model = LogisticRegression().fit(train_x, train_y)

#Make predictions and calculate acc, sens, specificity, f1 score and beta scores for them
y_pred = log_model.predict(test_x)

tn, fp, fn, tp = confusion_matrix(test_y, y_pred).ravel()
parameters = {'accuracy' : accuracy_score(test_y, y_pred),
							'sensitivity' : recall_score(test_y, y_pred),
							'speciality' : tn / (tn + fp)}
#Calculate F1 score
parameters['f_1_'] = f1_score(test_y, y_pred)