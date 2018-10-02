# coding: utf-8

# In[113]:


#get_ipython().magic('matplotlib notebook')
import matplotlib.pyplot as plt
import tweepy
from datetime import datetime
import pandas as pd
import json
import sys
from decimal import Decimal
sys.path.append("../")
import numpy as np
from config import (consumer_key, 
                    consumer_secret, 
                    access_token, 
                    access_token_secret)


# In[114]:


# Import and Initialize Sentiment Analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()


# In[115]:


# Setup Tweepy API Authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())


# In[116]:


# Target Search Term

target_names = ["@BBC","@CBS","@CNN","@FoxNews","@nytimes"]
sentiment_elements=["neg",'neu','pos','compound',"created_at",'text']
index2=pd.MultiIndex.from_product([[str(i) for i in range(100)], target_names],names=['#', 'Source'])

tweet_df=pd.DataFrame(columns=sentiment_elements,index=index2)

for name in target_names:
        public_tweets = api.user_timeline(screen_name=name,count=100,tweet_mode='extended')
        i_th=-1
        for tweet in public_tweets:
        # Run Vader Analysis on each tweet
            i_th+=1 
            result= analyzer.polarity_scores(tweet["full_text"])
            tweet_df.loc[(str(i_th),name),"compound"]=result["compound"]
            tweet_df.loc[(str(i_th),name),"pos"]=result["pos"]
            tweet_df.loc[(str(i_th),name),"neu"]=result["neu"]
            tweet_df.loc[(str(i_th),name),"neg"]=result["neg"]
            tweet_df.loc[(str(i_th),name),"created_at"]=datetime.strptime(tweet['created_at'], "%a %b %d %H:%M:%S %z %Y")
            tweet_df.loc[(str(i_th),name),"text"]=tweet["full_text"]
    
tweet_df['Tweets ago']=tweet_df.index.get_level_values(0)
tweet_df['Tweets ago']=pd.to_numeric(tweet_df['Tweets ago']);
tweet_df['compound']=pd.to_numeric(tweet_df['compound']);
tweet_df.head()
tweet_df=tweet_df.reset_index(level=[0,1])
tweet_df.head()


# In[117]:


##transform the sentiments into a pivotal table
tweet_df=tweet_df.sort_values(by="created_at")
seniment_data=tweet_df[["Source","compound","Tweets ago"]].pivot(index="Tweets ago",columns="Source",values="compound");


# ## The first plot

# In[118]:


fig1, ax1 = plt.subplots()
x_axis=seniment_data.reset_index()["Tweets ago"].values

s1=ax1.scatter(x_axis, seniment_data["@BBC"].values,marker='.',c='r',label="BBC",edgecolors='k',s=100);
s2=ax1.scatter(x_axis, seniment_data["@CBS"].values,marker='.',c='k',label="CBS",edgecolors='k',s=100);
s3=ax1.scatter(x_axis, seniment_data["@CNN"].values,marker='.',c='g',label="CNN",edgecolors='k',s=100);
s4=ax1.scatter(x_axis, seniment_data["@FoxNews"].values,marker='.',c='b',label="FoxNews",edgecolors='k',s=100);
s5=ax1.scatter(x_axis, seniment_data["@nytimes"].values,marker='.',c='y',label="nytimes",edgecolors='k',s=100);


#https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
# Shrink current axis by 20%
box = ax1.get_position()
pos=ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
leg=ax1.legend(loc='upper left', bbox_to_anchor=(1, 1.0))
plt.gca().invert_xaxis()

from datetime import date
title=ax1.set_title(f"Sentimetal analysis of media tweets({date.today().strftime('%d/%m/%y')})");
xlabel=ax1.set_xlabel("Tweets Ago")
ylabel=ax1.set_ylabel("Tweet Polarity")
plt.savefig('Sentimetal_data_of_media_tweets.png')


# ##  The second plot 

# In[119]:


fig2, ax2 = plt.subplots()
bar=seniment_data.mean().plot(kind='bar',ax=ax2,rot=0)
xtick_label=ax2.set_xticklabels(["BBC","CBS","CNN","Fox","NYT"])
xlabel=ax2.set_xlabel("")
ylabel=ax2.set_ylabel("Tweet Polarity")
title=ax2.set_title(f"Overall media sentiment based on twitter({date.today().strftime('%d/%m/%y')})")

def labelbar(x,y,t):
    txt=ax2.text(x, y,t,ha='center', va='bottom')

x=ax2.get_xticks()
y=4*[-0.03]+[0.03]
t = [round(Decimal(seniment_data.mean().values[i]),2) for i in range(len(x))]

for i in range(len(x)):
    labelbar(x[i],y[i],str(t[i]))
# Save the Figure
plt.savefig('Overall_media_sentiment.png')


# ## save DataFrame  to csv

# In[120]:


seniment_data.to_csv("sentiment_data.csv")
tweet_df.to_csv("full_sentiment_data.csv")


# ## Observations:
# > -  the twittering frequency of CBS is the lowest among the 5 organzations. If we take most recent 100 tweets, the timestamp goes back to early August. For BBC, it goes back to 2018-09-25. For CNN, FoxNews and NewYorkTimes, about half an day ago.
# -  For the past 100 tweets, NewYorkTimes is the only one that has overall negative mood and the FoxNews avearges almost to neutral.
# -  The sentiment of each organization fluctuates greatly. A great portion of tweets for all organizations are neutral. 
# -  During the past half day, FoxNews changes from negative to positive while NewYorkTimes changes from weakly positive to very negative. 
# -  During the past 2 months, CBS changes from very positive to positive.
# - During the past week, BBS changes its mood from  positive to weakly positive.
# 

# In[121]:


tweet_df[tweet_df["Source"]=="@nytimes"]

