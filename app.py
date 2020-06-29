import sys
import os
import csv
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import xgboost as xb

def get_input():
	filename = sys.argv[1]
	f = open(filename, 'r')
	csv_file = pd.read_csv(f)
	return csv_file 

def get_choice(df):
	choice = 0 
	while(choice!=5):
		print('Enter a choice')
		print('1 For Shape')
		print('2 For Description')
		print('3 For Column Names')
		print('4 For Information')
		print('5 For Exit')
		choice = input('') 
		if choice == '1':
			print(df.shape)
		elif choice == '2':
			print(df.describe())
		elif choice == '3':
			print(df.columns)
		elif choice == '4':
			print(df.info())
		elif choice == '5':
			exit_program()
			break
		else:
			print('Wrong Choice')

def show_columns(df):
	obj_cols = df.select_dtypes(include=['object']).columns
	num_cols = df.select_dtypes(include=['number']).columns
	print('Categorical Features are')
	print(obj_cols)
	print('\n')
	print('Numerical Features are')
	print(num_cols)


def show_null(df):
	print(df.isnull().sum())

def fill_null(df):
	obj_cols = df.select_dtypes(include=['object']).columns
	num_cols = df.select_dtypes(include=['number']).columns
	for i in num_cols:
		df[i]=df[i].fillna(df[i].mean())
	for i in obj_cols:
		df[i]=df[i].fillna(df[i].mode()[0])
		df[i]=pd.gets_dummies(df[i])
	print('After filling null values')
	print(df.isnull().sum())
	return df

def plot_graph(df):
	sns.heatmap(df.corr(),linewidths=0.5,annot=True)
	plt.show()

def ask_choice():
	choice1 = 0 
	print('1 For Classification')
	print('2 For Regression')
	choice1 = input('')
	return choice1

def target_feature():
	print('Enter target feature')
	feature = input('')
	return feature

def get_top_features(df,feature):
	X = df.drop(feature,axis=1)
	y = df[feature]
	clf = ExtraTreesClassifier(n_estimators=50)
	clf = clf.fit(X, y)
	feature_importance = clf.feature_importances_ 
	for i in range(0,len(df.columns)-1):
		print(df.columns[i],feature_importance[i])

def preprocessing(null_filled_df):
	obj_cols = df.select_dtypes(include=['object']).columns
	for i in obj_cols:
		null_filled_df[i]=pd.Factorize(null_filled_df[i])[0]
	return null_filled_df
	print('Successfully preprocessed')

def perform_model(choice1,preprocessed_df,feature):
	if choice1=='1':
		X = df.drop(feature,axis=1)
		y = df[feature]
		train_X , test_X , train_y , test_y = train_test_split(X,y,test_size=0.2,random_state=0)
		clf1=RandomForestClassifier()
		clf1.fit(train_X,train_y)
		clf1_score= clf1.score(test_X,test_y)

		clf2=xb.XGBClassifier()
		clf2.fit(train_X,train_y)
		clf2_score = clf2.score(test_X,test_y)
		max_clf_score = max(clf1_score,clf2_score)
		return max_clf_score

	if choice1=='2':
		X = df.drop(feature,axis=1)
		y = df[feature]
		train_X , test_X , train_y , test_y = train_test_split(X,y,test_size=0.2,random_state=0)

		reg1 = LogisticRegression()
		reg1.fit(train_X,train_y)
		reg1_score = reg1.score(test_X,test_y)

		reg2 = RandomForestRegressor()
		reg2.fit(train_X,train_y)
		reg2_score = reg2.score(test_X,test_y)
		max_reg_score = max(reg1_score,reg2_score)
		return max_reg_score, model_name

def graph_choice(df):
	graph_choice = 0 
	graph_df = df
	while(choice!=4):
		print('1 Histogram')
		print('2 Distplot')
		print('3 Countplot')
		print('4 Exit')
		graph_choice = input('')
		if graph_choice == '1':
			plot_hist(graph_df)
		if graph_choice == '2':
			plot_dist(graph_df)
		if graph_choice == '3':
			plot_count(graph_df)
		if graph_choice =='4':
			exit_program()
			break

def plot_hist(graph_df):
	print('Enter feature')
	feature = input('')
	plt.hist(df[feature])
	plt.show()


def plot_dist(graph_df):
	print('Enter feature')
	feature = input('')
	sns.distplot(df[feature])
	plt.show()

def plot_count(graph_df):
	print('Enter feature1')
	print('Enter feature2')
	feature1 = input('')
	feature2 = input('')
	sns.countplot(x=df[feature1],hue=df[feature2])
	plt.show()

def exit_program():
	print('Exit successful')

if __name__ == '__main__':
	df = get_input()
	print('File input successful')
	get_choice(df)
	print('\n')
	show_columns(df)
	feature = target_feature()
	print('\n')
	show_null(df)
	print('\n')
	null_filled_df = fill_null(df)
	print('1 for Heatmap')
	print('2 for Exit')
	choice = input('')
	if choice == '1':
		plot_graph(df)
	else:
		exit_program()

	graph_choice(df)
	preprocessed_df = preprocessing(null_filled_df)
	print('Feature importances')
	print('\n')
	get_top_features(df,feature)
	print('\n')
	choice1=ask_choice()
	score = perform_model(choice1,preprocessed_df,feature)
	print('Accuracy')
	print(score)
