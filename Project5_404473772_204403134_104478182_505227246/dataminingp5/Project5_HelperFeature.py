import json
import pytz
import datetime
import pandas as pd
import numpy as np

def extract_feat(file_name, flag = 0, windowsize = 3600):

    pst_tz = pytz.timezone('America/Los_Angeles')
    # to find the time of first and last post in the set
    mincit = 10e9
    maxcit = -1


    with open(file_name,encoding="utf-8") as f:
        for line in f:
            json_object = json.loads(line)
            if json_object['citation_date'] < mincit:
                mincit = json_object['citation_date']
            if json_object['citation_date'] > maxcit:
                maxcit = json_object['citation_date']
    # found the time of first and last post (absolute time)


    # rounding the beginstamp to the correct full first dataset
    # legit features must be rich in features (begin) AND target (end)

    beginstamp = mincit
    if beginstamp % windowsize == 0:
        beginstamp = beginstamp
    else:
        beginstamp -= beginstamp % windowsize
        beginstamp = beginstamp + windowsize
        feat_begin_hr = int(beginstamp / windowsize)


    endstamp = maxcit
    if endstamp % windowsize == 0:
        endstamp = endstamp - windowsize
    else:
        endstamp -= endstamp % windowsize
        endstamp = endstamp - windowsize
        feat_end_hr = int(endstamp / windowsize)


    #print((endstamp-beginstamp)/windowsize)

    #initialize the VARs
    if flag == 0:
        number_of_tweets = [0 for stamp in range(feat_begin_hr,feat_end_hr)]
        number_of_retweets = [0 for stamp in range(feat_begin_hr,feat_end_hr)]
        s_number_of_followers = [0 for stamp in range(feat_begin_hr,feat_end_hr)]
        max_of_followers = [0 for stamp in range(feat_begin_hr,feat_end_hr)]
        time_of_day = [0 for stamp in range(feat_begin_hr,feat_end_hr)]
        target = [0 for stamp in range(feat_begin_hr,feat_end_hr)]

        for idx, stamp in enumerate(range(beginstamp,endstamp,windowsize)):
            time_of_day[idx] = datetime.datetime.fromtimestamp(stamp, pst_tz).hour


        with open(file_name, encoding="utf-8") as f:
            for line in f:
                # print(line)
                json_object = json.loads(line)
                stamp = json_object['citation_date']
                stamp -= stamp % windowsize
                idx = int((stamp-beginstamp)/windowsize)
                #print(idx)

                if idx >= 0 and idx < feat_end_hr - feat_begin_hr:
                    number_of_tweets[idx] += 1
                    number_of_retweets[idx] += json_object['metrics']['citations']['total']
                    s_number_of_followers[idx] += json_object['author']['followers']
                    max_of_followers[idx] = max(max_of_followers[idx],json_object['author']['followers'])

                if idx > 0 and idx < feat_end_hr - feat_begin_hr + 1:
                    target[idx-1] += 1

        table = [number_of_tweets, number_of_retweets, s_number_of_followers, max_of_followers, time_of_day, target]
        df = pd.DataFrame(table)
        df = df.transpose()
        df.columns= ['N','NR','SNF','MF','TD','target']

    else:
        number_of_tweets = [0 for stamp in range(feat_begin_hr,feat_end_hr)]
        number_of_retweets = [0 for stamp in range(feat_begin_hr,feat_end_hr)]
        s_number_of_followers = [0 for stamp in range(feat_begin_hr,feat_end_hr)]
        max_of_followers = [0 for stamp in range(feat_begin_hr,feat_end_hr)]
        time_of_day = [0 for stamp in range(feat_begin_hr,feat_end_hr)]

        number_of_impressions = [0 for stamp in range(feat_begin_hr,feat_end_hr)]
        ranking_score = [0 for stamp in range(feat_begin_hr,feat_end_hr)]
        user_count_friends = [0 for stamp in range(feat_begin_hr,feat_end_hr)]
        listed_count = [0 for stamp in range(feat_begin_hr,feat_end_hr)]
        s_foll_of_orig_auth = [0 for stamp in range(feat_begin_hr,feat_end_hr)]



        target = [0 for stamp in range(feat_begin_hr,feat_end_hr)]

        for idx, stamp in enumerate(range(beginstamp,endstamp,windowsize)):
            time_of_day[idx] = datetime.datetime.fromtimestamp(stamp, pst_tz).hour


        with open(file_name, encoding="utf-8") as f:
            for line in f:
                # print(line)
                json_object = json.loads(line)
                stamp = json_object['citation_date']
                stamp -= stamp % windowsize
                idx = int((stamp-beginstamp)/windowsize)
                #print(idx)

                if idx >= 0 and idx < feat_end_hr - feat_begin_hr:
                    number_of_tweets[idx] += 1
                    number_of_retweets[idx] += json_object['metrics']['citations']['total']
                    s_number_of_followers[idx] += json_object['author']['followers']
                    max_of_followers[idx] = max(max_of_followers[idx],json_object['author']['followers'])

                    number_of_impressions[idx] += json_object['metrics']['impressions']
                    ranking_score[idx] += json_object['metrics']['ranking_score']
                    user_count_friends[idx] += json_object['tweet']['user']['friends_count']

                    hehe = json_object['tweet']['user']['listed_count']
                    listed_count[idx] += hehe if hehe != None else 0

                    s_foll_of_orig_auth[idx] += json_object['original_author']['followers']



                if idx > 0 and idx < feat_end_hr - feat_begin_hr + 1:
                    target[idx-1] += 1

        table = [number_of_tweets, number_of_retweets, s_number_of_followers, max_of_followers, time_of_day,\
        number_of_impressions, ranking_score, user_count_friends, listed_count, s_foll_of_orig_auth, target]
        df = pd.DataFrame(table)
        df = df.transpose()
        df.columns= ['N','NR','SNF','MF','TD','NI','RS','UCF','LC','FOA','target']


    # print(df)
    return df

if __name__ == '__main__':
    tweet_gohawks = extract_feat("tweets_#nfl.txt", flag = 1)
    print(tweet_gohawks)


