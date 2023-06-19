#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import needed modules
import json
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

from gensim import corpora
from gensim.models import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel

import networkx as nx
from nltk.metrics import jaccard_distance


# In[3]:


#Load data
with open('tweets_clean','r') as file:
    tweets_clean = json.load(file)

#Function that mconverts a list of lists to one single list
def flatten_list(l):
    flat_list = [item for sublist in l for item in sublist]
    return flat_list
tweets_bag = flatten_list(tweets_clean) #put the dataset's tokens in one list


# ## Topic Extraction

# In[4]:


#Randomness is controlled for reproduciability of the results
random_seed = 2
np.random.seed(random_seed)


# In[5]:


#Map tokens to integer id's
dictionary = corpora.Dictionary(tweets_clean)
#create BoW of all tweets
corpus = [dictionary.doc2bow(tweets_bag)]
coherence_list = []


# In[6]:


#GRID SEARCH: Find optimal amount of topics
for n in range(1,11):
    lda = LdaMulticore(corpus=corpus, num_topics=n, id2word=dictionary, eta='auto', random_state=random_seed)
    cm = CoherenceModel(model=lda, texts=tweets_clean, dictionary=dictionary, coherence='c_v')
    coherence = cm.get_coherence()
    coherence_list.append(coherence)


# In[39]:


#Plot coherence scores
plt.xlabel('number of topics')
plt.ylabel('coherence score (C_v)')
#plt.title('LDA performance', fontsize=16)
plt.xticks(range(1,11))
plt.grid(color='w', linestyle='solid')
plt.tick_params(colors='gray', direction='out')
plt.plot(range(1,11), coherence_list, color='#468499')
plt.savefig("LDA_gridsearch.pdf")
plt.rcParams['font.size'] = 14
plt.show()


# In[33]:


#Run the best model
n_topics = 5 #manually select num_topics
lda = LdaMulticore(corpus=corpus, num_topics=n_topics, id2word=dictionary, eta='auto', random_state=2)


# In[34]:


topics = {}
termlist = []
#concentrate the 30 most relevant terms for each topic
for topic in range(n_topics):
    topics[topic] = []

    for term in lda.get_topic_terms(topicid=topic, topn=30):
        word = dictionary[term[0]] #find relevant word-term
        termlist.append(word)  #add every term in a list
        topics[topic].append(word) #group terms by topic in a dictionary as well

termlist = list(set(termlist)) #keep only unique terms


# ## Tweets Segmentation

# In[35]:


#Distribute tweets among the discovered topics
topic_dist = []

#find the most probable topics for each tweet
for tweet in tweets_clean:
    bow = [dictionary.doc2bow(tweet)]
    dist = lda.get_document_topics(bow=bow, minimum_probability=0.2)[0] #topics with prob lower than 20% will be ignored
    topic_dist.append(dist) #tuple containing the topic and its probability

#assign same topic tweets in the same cluster
clusters = {}
for n in range(1,n_topics+1):
    clusters[n] = []

for id, tweet in enumerate(topic_dist):
    for topic in tweet:
        cluster = topic[0] + 1 #transform topic names from 0-4 to 1-5
        clusters[cluster].append(id)  #add the tweet id to the right cluster(s).


# In[36]:


#Count how many tweets occurred in more than one clusters
count = 0
for i in topic_dist:
    if len(i) > 1:  #if it contains more than one tuple
        count += 1
print(count)


# ## Visualizations: Topic frequency and measuring overlap

# In[40]:


#Topic Population frequency statistics
cluster_frequency = {key: len(value) for key, value in clusters.items()} #get the length
cluster_names = list(cluster_frequency.keys())
frequencies = list(cluster_frequency.values())

# Plot
plt.grid(False)
plot = plt.bar(cluster_names, frequencies, color='#468499')

#add the frequency counts on top of the bars
for bar in plot:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.0f}', ha='center', va='bottom', color='gray')

plt.xlabel('n Topic')
plt.ylabel('num of tweets')
#plt.title('Topic frequency', fontsize=16)
plt.ylim(bottom=min(frequencies)/5)
plt.tick_params(colors='gray', direction='out')
plt.xticks(np.arange(1,n_topics+1))
plt.savefig("topic_population.pdf")
plt.rcParams['font.size'] = 14
plt.show()


# In[41]:


cluster_tokens = []
#Group together tweet tokens from the same topics
for topic in range(len(clusters)):
    temp = []
    for tweet_id in clusters[topic+1]:
        tweet = tweets_clean[tweet_id]
        temp.append(tweet)
    temp = flatten_list(temp)
    cluster_tokens.append(temp) #list of 5 sublists, containing each topic's total tokens

freq_per_topic = {}
for word in termlist:
    freq_per_topic[word] = []

#each topic's frequency for each relevant word-term
for n_clusters in range(len(cluster_tokens)):
    for word in termlist:
        freq = cluster_tokens[n_clusters].count(word) #find freq count
        freq_scaled = freq/ len(cluster_tokens[n_clusters]) #divide by the number of words, since clusters are imbalanced
        freq_percentage = freq_scaled * 100 #convert percentage form
        freq_norm = round(freq_percentage,2) #round
        freq_per_topic[word].append(freq_norm) #dictionary with key: term and value: list of 5 frequency percentages

#list of the least informative relevant words (by eye-balling)
remove_list = ['day','many','year','back','time','get',
               'week','one','would','month','say','u',
               'want','still','today','like','need','know']

for word in remove_list:
    del freq_per_topic[word]


# In[42]:


#make a shorter version of termlist with only the most informative words
termlist_stripped = [word for word in termlist if word in freq_per_topic.keys()]
len(termlist_stripped)


# In[90]:


# Plot the term frequencies per topic
plt.figure(figsize=(10, 6))
plt.grid(False)
colors = ['#468499','#6C9DB6','#8BAFCC','#A9CDDF','#C7DFE9']

width = 0.2

for topic in range(n_topics):
    frequencies = [freq_per_topic[word][topic] for word in termlist_stripped]
    term = np.arange(len(termlist_stripped))
    plot = plt.bar(term + topic * width, frequencies, width=0.2, label=f'Topic {topic}', color=colors[topic], edgecolor="none")

    for bar in plot:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height, topic+1, ha='center', va='bottom', color='gray', fontsize=8)


plt.xlabel('Relevant terms', fontsize=17)
plt.ylabel('% occurence', fontsize=17)
plt.legend(range(1,n_topics+1), loc='upper right', fontsize=14)
#plt.title('Term Frequency by Topic')

plt.xticks(term + width * (n_topics -1) / 2, termlist_stripped, fontname='Arial', rotation=80, fontsize=16)
plt.tick_params(axis='both', length=5, width=1, colors='gray', direction='out', bottom=True, left=True, size=5)
plt.subplots_adjust(bottom=0.3)

plt.savefig("term_frequency.pdf")
plt.show()


# In[124]:


#initialize empty matrices
co_occurrence_matrix = np.zeros((len(cluster_names), len(cluster_names))) #length of nameslist is the same as the number of topics (5)
jaccard_coef = np.zeros((len(cluster_names), len(cluster_names)))
edge_labels = {}

#Calculate co-occurence and jaccard similarity matrices
for i in range(5):
    for j in range(5):
        #jaccard similarity = 1- jaccard distance
        jaccard_coef[i][j] = round(1 - jaccard_distance(set(clusters[i+1]), set(clusters[j+1])),2)
        #clusters -> dict where {n_topic : [tweetIDs]}
        if i < j:
            edge_labels[(i,j)] = jaccard_coef[i][j] # dict where {(topici, topicj): coef}
        for tweet_id in clusters[i+1]:
            if tweet_id in clusters[j+1]:
                co_occurrence_matrix[i][j] += 1 #increase co-occurence score


# In[125]:


#Dataframe conversions
co_occurence_df = pd.DataFrame(co_occurrence_matrix)
co_occurence_df.columns = range(1, co_occurence_df.shape[1] + 1)  # Change column names
co_occurence_df.index = range(1, co_occurence_df.shape[0] + 1)  # Change row names

jaccard_df = pd.DataFrame(jaccard_coef)
jaccard_df.columns = range(1, jaccard_df.shape[1] + 1)  # Change column names
jaccard_df.index = range(1, jaccard_df.shape[0] + 1)  # Change row names

co_occurence_df.to_csv("support_weights.csv")
jaccard_df.to_csv("jaccard.csv")


# In[126]:


co_occurence_df


# In[24]:


jaccard_df


# In[157]:


#Network Graph
# create graph object from co-occurence matrix
graph = nx.from_numpy_array(np.array(jaccard_coef))
node_labels = {0: 'healthcare', 1:'coping', 2:'testing', 3:'death rates', 4:'socio-politics'}
layout = nx.kamada_kawai_layout(graph)

#plot topic nodes close to each other according to their jaccard score
nx.draw(graph, layout, with_labels=True, labels=node_labels, node_size=400, node_color='#468499', edge_color='lightgray', font_size=14)

#add the jaccard labels
nx.draw_networkx_edge_labels(graph, layout, edge_labels=edge_labels, label_pos=0.57, font_size=12)

#plt.title("Topic overlap complemented with jaccard coeff")
plt.savefig("topic_overlap.pdf")
plt.show()


# ## Association Rule Mining

# In[26]:


#Calculate confidence for each topic pair
confidence_weights =  np.zeros((5,5))

for i in co_occurence_df:
    for j in co_occurence_df:
        support_i = co_occurence_df[i][i]
        support_i_j = co_occurence_df[i][j]
        confidence = support_i_j / support_i
        confidence_weights[i-1][j-1] = confidence

confidence_df = pd.DataFrame(confidence_weights)
confidence_df.columns = range(1, confidence_df.shape[1] + 1) #rename columns to match cluster names
confidence_df.index = range(1, confidence_df.shape[0] + 1)  #rename rows to match cluster names
confidence_df.to_csv("confidence_weights.csv")
confidence_df

