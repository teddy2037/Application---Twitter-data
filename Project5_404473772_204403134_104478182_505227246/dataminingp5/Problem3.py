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
import statsmodels.api as sm

pst_tz = pytz.timezone('America/Los_Angeles')

names = ["tweets_#gohawks.txt", "tweets_#gopatriots.txt", \
         "tweets_#nfl.txt", "tweets_#patriots.txt", "tweets_#sb49.txt", "tweets_#superbowl.txt"]
feat = ["number of tweets","number of retweets", "sum of followers", "max of followers",\
        "time of day", "number of impressions", "ranking score", "friend count of user", "listed count", "foll. of orig. author"]

def LR(datum):
    reg_fin = LinearRegression().fit(datum[:,:-1], datum[:,-1])
    pred = reg_fin.predict(datum[:,:-1])
    mse_trn = (mean_squared_error(datum[:, -1], pred))
    r2 = r2_score(datum[:, -1], pred)
    return mse_trn,r2

def TP(datum, j):
    A = sm.add_constant(datum[:,:-1])
    model = sm.OLS(datum[:,-1],A)
    results = model.fit()
    t = results.tvalues
    p = results.pvalues
    for i in range(1, len(t)):
        print(str(feat[i-1]) + str("&") + str(round(t[i],4)) + str("&") + str(round(p[i],4)) + str("\\\\ \\hline"))


if __name__ == '__main__':
    print
    '''
    for i in range(0,6):
        p = extract_feat(names[i])
        a,b = LR(p.values)
        print("MSE and r2_score:\t" + str(a) + ", " + str(b))
    print
    print("Writing t- and p- values:")
    '''
    print('\n')
    for i in range(0,6):
        p = extract_feat(names[i])
        TP(p.values, i)
        print('\n')

