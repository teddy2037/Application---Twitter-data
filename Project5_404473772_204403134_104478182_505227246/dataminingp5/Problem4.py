from Project5_HelperFeature import extract_feat 
import json
import datetime
from math import ceil
import datetime
import pytz
import numpy as np
import dateutil.parser
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from Problem3 import LR, TP
import statsmodels.api as sm

pst_tz = pytz.timezone('America/Los_Angeles')

names = ["tweets_#gohawks.txt", "tweets_#gopatriots.txt", \
         "tweets_#nfl.txt", "tweets_#patriots.txt", "tweets_#sb49.txt", "tweets_#superbowl.txt"]


if __name__ == '__main__':
    '''
    for i in range(0,6):
        p = extract_feat(names[i],flag = 1)
        a,b = LR(p.values)
        print("MSE and r2_score:\t" + str(a) + ", " + str(b))
    print
    print("Writing t- and p- values:")
    '''
    print('\n')
    for i in range(0,6):
        p = extract_feat(names[i], flag = 1)
        TP(p.values, i)
        print('\n')

