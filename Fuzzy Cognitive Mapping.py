#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import needed modules
import os
import pandas as pd
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#Load FCM weights
path = os.getcwd() + "/confidence_weights.csv"
weights = pd.read_csv(path)
weights = weights.drop(['Unnamed: 0'], axis=1) #drop the extra column: the FCM does not need it

w0 = np.asarray(weights)
#round the weights for readability
w0_rounded = [[round(element, 3) for element in row] for row in w0]
#convert to array
w0 = np.asarray(w0_rounded)
print(w0)


# In[3]:


#Define Transfer function and Reasoning Rule

"""Rescaled transfer function"""
def rescaled(A):
    if la.norm(A)==0.0:
        return np.zeros(A.shape)
    else:
        return A / la.norm(A)

"""Recurrent reasoning process"""
def reasoning(W, A, T=50, phi=0.8, function=rescaled):

    states = np.zeros([len(A), T, len(W)])
    states[:,0,:] = A

    for t in range(1,T):
        A = states[:,t,:] = (phi * function(np.matmul(A, W)) + (1-phi) * states[:,0,:])

    return states


# In[6]:


#Discretize the continuous activations
low = 'low'
high = 'high'
random = 'random'

#this function outputs a random activation (float) from 3 uniform distributions
def act(degree):
    activation = {'low': np.random.uniform(0.0, 0.5),
                  'high': np.random.uniform(0.7, 1.0),
                  'random': np.random.uniform(0.0, 1.000000001)}
    return activation[degree]


# In[109]:


#main simulations function
fontsize = 20
def simulation(activations=[0,0,0,0,0], f=rescaled, w0=w0,
               k=20, phi_values=[0.2,0.6,0.8,1.0],
               reps=1, name='fig'):
    """
    :param activations: list of 5 initial activation values for each node

    :param f: string, predefined transfer function

    :param w0: 5x5 array of int or floats, the weight matrix

    :param k: int, number of iterations

    :param phi_values: list of 4 floats, from 0 to 1 (inclusive), determines the degree of non-linearity

    :param reps: positive int, number of replications of the simulation. reps>1 is recommended only when activation values need to be randomly sampled

    :param phi_values: list of 4 floats from 0 to 1 (inclusive), phi value(s) that determine the nonlinearity of the model.
    phi=0: strong influence of initial activations, phi=1: min influence

    :param name: string, the name of the saved figure file

    :return: a matplotlib figure containing 4 subplots. One simulation plot for every value of phi. a file of the figure is also saved in the current directory
    """

    fig, ax= plt.subplots(2, 2, sharex=True, sharey=True, figsize=(14, 6))
    grid = [ax[0,0], ax[0,1], ax[1,0], ax[1,1]]

    for i in range(len(phi_values)):
        df = pd.DataFrame(columns=["feature","iteration","value"])

        for rep in range(reps):
            if reps > 1:
                #if simulation is replicated, randomly initialize the activations in each replication
                activations = [act(random), act(random), act(random), act(random), act(random)]

            for iter in range(k):

                A = np.array([activations])
                state = reasoning(w0, A, phi=phi_values[i], T=21, function=f)

                data_topic1 = state[0,:,0]
                data_topic2 = state[0,:,1]
                data_topic3 = state[0,:,2]
                data_topic4 = state[0,:,3]
                data_topic5 = state[0,:,4]

                data_topics = [data_topic1, data_topic2, data_topic3, data_topic4, data_topic5]
                topic_names = ['Healthcare', 'Coping', 'Testing', 'Death rates', 'Sociopolitics']
                df_per_topic = []

                #store the update steps of each feature at each iteration
                for topic_id in range(len(activations)):
                    dft = pd.DataFrame(columns=["feature","iteration","activation"])
                    data_topic = data_topics[topic_id]
                    dft["iteration"] = range(len(data_topic))
                    dft["value"] = data_topic.tolist()
                    dft["feature"] = topic_names[topic_id]
                    df_per_topic.append(dft)

                #add all dataframes together
                concatenate = [dft for dft in df_per_topic]
                concatenate.append(df)
                df = pd.concat(concatenate, ignore_index=True)

        #plot
        ax1 = sns.lineplot(data=df, x="iteration", y="value", hue="feature", ax=grid[i],
                           linewidth = 2.5, marker='o', markeredgecolor='None', errorbar='sd')
        ax1.xaxis.get_major_locator().set_params(integer=True)

        ax1.set_title('phi=' + str(phi_values[i]), fontsize=fontsize)
        ax1.set_xlabel('iteration', fontsize=fontsize)
        ax1.set_ylabel('activation', fontsize=fontsize)
        ax1.set_ylim([0, 1])
        ax1.legend().remove()


    plt.xticks(fontsize=fontsize)
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol = 5, loc='upper center', fontsize=fontsize)
    fig.tight_layout(rect=[0, 0.03, 1, 0.90], h_pad=0.5)
    sns.despine()
    plt.savefig(f"{name}.pdf")


# In[162]:


#Activating Topic 1 with high activation values
activations = [act(high), act(low), act(low), act(low), act(low)]
simulation(activations=activations, name='actTopic1')


# In[130]:


#Activating Topic 3 with high activation values
activations = [act(low), act(low), act(high), act(low), act(low)]
simulation(activations=activations, name='actTopic3')


# In[148]:


#Activating Topics  5 with high activation values
activations = [act(low), act(low), act(low), act(low), act(high)]
simulation(activations=activations, name='actTopic5')


# ## Randomized activations

# In[163]:


simulation(reps=50, name='actRandom')


# ## phi=1: Presence Proportions

# In[7]:


# Topic labels
labels = ['Healthcare', 'Coping Mechanisms', 'Testing', 'Death rates', 'Socio-politics']
#Topic presence for phi=1
sizes = [0.097, 0.475, 0.28, 0.81, 0.15]
colors = ['#468499','#6C9DB6','#8BAFCC','#A9CDDF','#C7DFE9']

# Create a pie chart of the topic's presence for phi=1
plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, textprops={'fontsize': 13})
#plt.title('Topic domination in the fixed-point attractor')
plt.savefig("topic_domination.pdf")
plt.show()

