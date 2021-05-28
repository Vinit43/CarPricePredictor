import pandas as pd

import seaborn as sns

from seaborn import *

import matplotlib.pyplot as plt

from matplotlib.pyplot import figure , show

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

def main():
	dataset = pd.read_csv("car.csv")
	print(dataset.head())
	
	print(dataset.shape)

	print(dataset["Transmission"].unique())

	print(dataset["Owner"].unique())
	print(dataset.describe())
	print(dataset.columns)
	dataset.drop(["Car_Name"], axis=1, inplace= True)
	print(dataset.head())
	print(dataset.columns)

	dataset["Current"]  = 2020
	dataset["NoOfYears"] = dataset["Current"] - dataset["Year"]
	print(dataset.head())

	dataset.drop(["Year"], axis=1, inplace= True)
	dataset.drop(["Current"], axis=1, inplace= True)
	print(dataset.head())


	dataset = pd.get_dummies(dataset , drop_first = True)
	print(dataset.head())

	# sns.pairplot(dataset)
	# show()

	# corrmat = dataset.corr()
	# top_corr_features = corrmat.index
	# plt.figure(figsize=(20,20))
	# g = sns.heatmap(dataset[top_corr_features].corr(),annot=True,cmap = "RdYlGn")
	# show()

	X = dataset.iloc[:,1:]

	Y = dataset.iloc[:,0]


	#Feature Importance

	model = ExtraTreesRegressor()
	model.fit(X,Y)
	print(model.feature_importances_)

	#plot graph of feature importances for better visualization
	feat_importances = pd.Series(model.feature_importances_, index=X.columns)
	feat_importances.nlargest(5).plot(kind='barh')
	plt.show()

	X_train , X_test , Y_train , Y_test = train_test_split( X , Y , test_size = 0.2 )

	#HyperParameters
	import numpy as np

	n_estimators = [int(x) for x in np.linspace(start = 100 , stop = 1200 ,num = 12 )]
	print(n_estimators)


	max_features = ["auto" , "sqrt"]

	max_depth = [int(x) for x in np.linspace(5,30 , num=6)]

	min_samples_split = [2,5,10,15,100]

	min_samples_leaf = [1,2,5,10]

	from sklearn.model_selection import RandomizedSearchCV

	random_grid = {'n_estimators': n_estimators,'max_features': max_features,'max_depth': max_depth,'min_samples_split' : min_samples_split,'min_samples_leaf': min_samples_leaf}

	print(random_grid)

	rf = RandomForestRegressor()

	rf_random = RandomizedSearchCV(estimator = rf , param_distributions = random_grid , scoring = 'neg_mean_squared_error' , n_iter = 10 ,cv =5 ,verbose =2 , random_state=42 , n_jobs = 1 )

	rf_random.fit(X_train , Y_train)

	predictions = rf_random.predict(X_test)

	# sns.distplot(Y_test - predictions)


	import pickle

	file = open('random_forest_regression_model.pkl' , 'wb')

	pickle.dump(rf_random , file)



if __name__ == "__main__":
	main()
