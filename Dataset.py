#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Import packages
import os
import pandas as pd
import snscrape.modules.twitter as sntwitter


# In[ ]:


#query specifies tweets with:
# the given hashtags,
# 20 minimum likes,
# english language,
# date range,
# and filters out reply tweets or ones that include links

query ="(#covid19 OR #covid OR #coronavirus) min_faves:20 lang:en until:2020-09-30 since:2020-06-01 -filter:links -filter:replies"
tweets = []
limit = 10000

#Warning: datascrapping follows, uncomment to run. Will overwrite the dataset.
"""
for tweet in sntwitter.TwitterSearchScraper(query).get_items():
    if len(tweets) == limit:
        break
    else:
        tweets.append([tweet.date, tweet.username, tweet.content])
"""

df = pd.DataFrame(tweets, columns=['Date', 'User', 'Tweet'])
df.to_csv("covid19_tweets.csv")

