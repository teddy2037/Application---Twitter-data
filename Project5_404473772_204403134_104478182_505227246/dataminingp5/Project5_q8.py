import json
import pytz
import datetime
import time
import pandas as pd
import winsound
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from tempfile import mkdtemp
from shutil import rmtree
from Project5_q6 import feature_extraction, min_max_timestamps
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.externals.joblib import Memory
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from pandas import ExcelWriter
from pandas import DataFrame, read_csv


pst_tz = pytz.timezone('America/Los_Angeles')


frequency = 2500  # Set Frequency To 2500 Hertz
duration = 1000  # Set Duration To 1000 ms == 1 second

s = 'file_aggreg.txt'

names = ["tweets_#gohawks.txt", "tweets_#gopatriots.txt", \
         "tweets_#nfl.txt", "tweets_#patriots.txt", "tweets_#sb49.txt", "tweets_#superbowl.txt"]




l = min_max_timestamps(s)

beginstamp = l[0]
endstamp = l[1]

# v = datetime.datetime.fromtimestamp(beginstamp, pst_tz)
# print(v)
# time1 = int(time.mktime(datetime.datetime(v.year, v.month, v.day, v.hour, v.minute, v.second, v.microsecond, pst_tz).timetuple()))

print('1')
stamp1 = int(time.mktime(datetime.datetime(2015, 2, 1, 8, 0, 0, 0, pst_tz).timetuple()))
stamp2 = int(time.mktime(datetime.datetime(2015, 2, 1, 20, 0, 0, 0, pst_tz).timetuple()))

#feature1 = feature_extraction(beginstamp,stamp1,3600,s)
# feature2 = feature_extraction(stamp1,stamp2,300,s)
feature3 = feature_extraction(stamp2,endstamp,3600,s)

# features = feature_extraction(beginstamp,endstamp,3600,s)


print('2')


param_grid = {
'max_depth': [10, 20, 40, 60, 80, 100, 200, None],
'max_features': ['auto', 'sqrt'],
'min_samples_leaf': [1, 2, 4],
'min_samples_split': [2, 5, 10],
'n_estimators': [200, 400, 600, 800, 1000,
1200, 1400, 1600, 1800, 2000]
}

# Create a based model
rf = GradientBoostingRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV( estimator = rf, param_grid = param_grid,
                           cv=KFold(5, shuffle=True), n_jobs=-1,verbose = 2,scoring='neg_mean_squared_error')



dk1 = feature3.values
print('3')
# winsound.Beep(frequency, duration)
grid_search.fit(dk1[:,:-1], dk1[:,-1])

print(grid_search.best_params_)

df = pd.DataFrame(grid_search.cv_results_)

# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('regressortimestamp3.xlsx', engine='xlsxwriter')

# Convert the dataframe to an XlsxWriter Excel object.
df.to_excel(writer, sheet_name='Sheet1')

# Close the Pandas Excel writer and output the Excel file.
writer.save()

# winsound.Beep(frequency, duration)

# file = r'randomforest.xls'
# df = pd.read_excel(file)


