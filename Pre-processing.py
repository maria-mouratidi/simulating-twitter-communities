#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Import packages
import os
import re
import pandas as pd
import json

from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer


# In[3]:


path = os.getcwd() + "/covid19_tweets.csv"
raw_data = pd.read_csv(path)
raw_data = raw_data.drop(labels='Unnamed: 0', axis=1)
raw_data.head(5)


# In[4]:


#Tokenize with TweetTokenizer
tweet_tokenizer = TweetTokenizer()

#filter out @usernames, links and emojis and punctuation tokens
pattern = r"(?:\@[\w_]+|https?\://\S+|[\U00010000-\U0010ffff]|\w*[^\w\s]+\w*)"

tweets_tokenized = []
for sentence in raw_data.iloc[:, 2]: #select Tweet column
    tokens = tweet_tokenizer.tokenize(sentence)
    filtered_tokens = [re.sub(r'^#', '', token.lower()) for token in tokens if not re.match(pattern, token)]
    tweets_tokenized.append(filtered_tokens)


# In[5]:


#Remove stop words

stop_words = set(stopwords.words('english'))
stop_words.update(('covid', 'corona', 'coronavirus', 'covid19'))
tweets_stripped = []
for sentence in tweets_tokenized:
    tweets_stripped.append([w for w in sentence if (not w.lower() in stop_words and not w.isdigit())])


# In[6]:


tweets_stripped.remove(tweets_stripped[8048]) #this tweet was left empty after all the pre-processing, so it will be removed


# In[7]:


#Use a lemmatizer to remove variation

tweets_clean = []

for tweet in tweets_stripped:
    current_stemmed_tweet = []
    for word in tweet:
        stem = WordNetLemmatizer().lemmatize(word)
        current_stemmed_tweet += [stem]
    tweets_clean += [current_stemmed_tweet]

with open('tweets_clean','w') as file:
    json.dump(tweets_clean, file)


# In[8]:


#Rejoin each tokenized and preprocessed tweet into strings
tweets_strings = []
for tweet in tweets_clean:
    tweets_strings += [' '.join(tweet)]

len(tweets_strings)

with open('tweets_strings', 'w') as file:
    json.dump(tweets_strings, file)

