

```python
Your final output should provide a visualized summary of the sentiments expressed in Tweets sent out by the following news organizations: BBC, CBS, CNN, Fox, and New York times.

The first plot will be and/or feature the following:

Be a scatter plot of sentiments of the last 100 tweets sent out by each news organization, ranging from -1.0 to 1.0, where a score of 0 expresses a neutral sentiment, -1 the most negative sentiment possible, and +1 the most positive sentiment possible.
Each plot point will reflect the compound sentiment of a tweet.
Sort each plot point by its relative timestamp.
The second plot will be a bar plot visualizing the overall sentiments of the last 100 tweets from each organization. For this plot, you will again aggregate the compound sentiments analyzed by VADER.

The tools of the trade you will need for your task as a data analyst include the following: tweepy, pandas, matplotlib, seaborn, textblob, and VADER.
Pull last 100 tweets from each outlet.
Perform a sentiment analysis with the compound, positive, neutral, and negative scoring for each tweet.
Pull into a DataFrame the tweet's source acount, its text, its date, and its compound, positive, neutral, and negative sentiment scores.
Export the data in the DataFrame into a CSV file.
Save PNG images for each plot.
```


```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

import tweepy
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

```


```python
consumer_key = "Ed4RNulN1lp7AbOooHa9STCoU"
consumer_secret = "P7cUJlmJZq0VaCY0Jg7COliwQqzK0qYEyUF9Y0idx4ujb3ZlW5"
access_token = "839621358724198402-dzdOsx2WWHrSuBwyNUiqSEnTivHozAZ"
access_token_secret = "dCZ80uNRbFDjxdU2EckmNiSckdoATach6Q8zb7YYYE5ER"


auth = tweepy.OAuthHandler(consumer_key, consumer_secret) 
auth.set_access_token(access_token, access_token_secret) 
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())


```


```python
target_user = ['@BBCNews', '@CBSNews', '@CNN', '@FoxNews', '@NYT']
```


```python
compound_list = []  


for user in target_user:  
    
    for x in range(1,6):
        
        
        tweets = api.user_timeline(user, page=x) 
        
        for tweet in tweets:
            
            compound = analyzer.polarity_scores(tweet.text)["compound"]
            compound_list.append(compound)
        
```

    [-0.4391, 0.0, 0.0, -0.3182, 0.0, -0.3182, -0.6249, -0.875, -0.7579, 0.0, -0.296, -0.8225, -0.7717, 0.7003, 0.4215, 0.765, -0.5423, -0.5994, 0.0, -0.3818, 0.4019, -0.5423, -0.296, -0.6249, 0.0, 0.0, 0.0, 0.0, 0.6486, 0.4023, -0.2732, 0.6908, 0.5106, -0.5859, 0.0, -0.7855, -0.3818, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.4767, 0.0, -0.6124, 0.25, 0.0, -0.0516, 0.2023, -0.1027, 0.0, -0.9136, -0.765, -0.8807, 0.0, 0.4939, -0.0258, 0.3182, 0.4939, 0.3612, 0.2732, 0.4576, -0.25, -0.6124, -0.3818, 0.4215, -0.3818, -0.3818, -0.5106, -0.3818, -0.2263, -0.8481, 0.4939, 0.4215, 0.0, 0.0, 0.0, 0.3034, 0.0, 0.0, -0.1027, -0.4404, 0.1779, 0.0, 0.0258, -0.8176, -0.5423, 0.7907, -0.296, 0.4588, 0.5334, 0.0, 0.0, -0.6249, -0.4767, -0.4939, -0.296, -0.296, -0.4019, 0.0, -0.3182, 0.0, 0.0, 0.0, 0.0, 0.0, -0.296, -0.3182, 0.0, 0.6298, -0.8934, 0.4973, 0.1779, -0.6908, -0.6705, 0.0, -0.128, -0.5994, 0.2732, 0.0, 0.4588, -0.5994, 0.0, -0.5994, 0.0, 0.0, -0.6901, -0.7579, 0.1779, -0.2732, -0.296, 0.0, -0.6705, 0.3818, 0.0, 0.0, 0.3535, 0.2023, 0.0, 0.4215, -0.6597, -0.1686, 0.4404, 0.0, 0.0, -0.3506, 0.0, -0.3182, 0.0, -0.3818, -0.296, 0.0, 0.0, -0.6124, -0.9246, -0.0516, -0.1531, 0.1027, -0.5574, -0.4588, 0.296, -0.7579, -0.5095, 0.0, -0.4019, -0.9371, 0.0, -0.3612, 0.0, 0.5719, 0.0, 0.5994, 0.0, 0.0, -0.6486, -0.5574, 0.0, 0.0, 0.0, -0.3182, -0.4019, 0.3818, -0.0516, 0.0, -0.8625, 0.0, 0.0, -0.3818, -0.6486, 0.0, -0.6124, -0.5994, 0.0, 0.0, 0.5859, -0.2732, 0.296, 0.1027, 0.0, 0.4973, -0.4939, -0.7506, 0.0, -0.2023, -0.4939, 0.0, 0.6209, -0.34, 0.6369, -0.4019, 0.0, -0.4404, 0.7506, -0.5423, 0.0, -0.6908, 0.0, 0.1779, 0.0, 0.0772, 0.296, -0.6705, 0.4215, 0.0, 0.0, 0.4973, -0.4404, 0.1027, 0.0, 0.6209, 0.1531, 0.4019, 0.5719, -0.4215, -0.1779, 0.4939, 0.0, 0.0, -0.34, 0.0, -0.2023, 0.0, 0.0, -0.6908, -0.7579, -0.2263, -0.0772, -0.4939, 0.0, 0.717, 0.0, -0.34, -0.6486, -0.4588, 0.0, 0.3612, 0.0, 0.4404, 0.4588, -0.1531, 0.0772, 0.34, -0.5267, -0.3182, 0.4588, 0.5267, -0.8735, -0.4019, 0.2732, 0.0, -0.7269, 0.0, -0.1695, -0.4588, 0.0, 0.6486, -0.5106, 0.0, 0.0, 0.3612, -0.128, 0.2023, 0.0, 0.0, 0.0, -0.7845, 0.0, 0.0516, 0.0, -0.7096, 0.0, 0.6486, -0.5106, 0.0, -0.5994, 0.0, -0.4976, -0.2263, 0.0, 0.0, -0.2924, -0.4767, 0.0, -0.34, -0.3818, 0.0, 0.4939, -0.4939, -0.1531, 0.0, 0.3612, -0.6124, -0.7048, 0.0, -0.296, 0.1531, -0.079, 0.0, -0.7579, -0.4215, 0.0, 0.0, 0.1139, -0.6908, 0.0, 0.0, 0.0, 0.0, -0.1154, -0.7845, 0.0, 0.5423, -0.3818, -0.6705, 0.5719, 0.6908, 0.4585, -0.0516, 0.0, 0.2598, 0.0, 0.0, 0.6124, -0.3182, 0.0, -0.4767, 0.8519, 0.0, 0.3612, 0.128, -0.4588, -0.4404, -0.25, 0.4019, 0.395, -0.8519, 0.0, 0.0772, -0.5423, 0.5106, -0.8625, 0.507, -0.7003, 0.0, 0.0, 0.0, -0.4767, 0.34, 0.0, -0.4767, 0.2023, 0.0, 0.4939, 0.6124, 0.4973, -0.1053, 0.0, -0.5994, 0.0, 0.0, 0.0, -0.4215, 0.0, -0.8625, -0.34, -0.6908, -0.1779, 0.0, -0.6705, 0.0772, -0.5574, 0.4019, -0.6486, -0.5574, 0.4497, 0.0, 0.0, 0.0, -0.2263, 0.6124, 0.6908, 0.0, -0.34, -0.6124, 0.5106, 0.0, -0.3818, 0.4019, -0.743, 0.4019, -0.68, -0.1027, -0.5994, 0.1027, 0.0, 0.0, 0.4019, 0.1027, -0.5719, 0.4588, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5106, 0.0, 0.0, 0.4939, 0.5719, 0.296, -0.7783, 0.0, 0.0, 0.0, 0.1779, -0.4767, 0.0, 0.0, 0.0, 0.0, 0.4215, 0.0, 0.5106, 0.0, 0.0, 0.0, -0.4019, -0.34, 0.0, 0.3182, 0.6486, 0.1779, 0.0, 0.0, 0.0, 0.7003, 0.6369, 0.6652, -0.743, 0.5106, -0.4588, -0.5106, -0.4215, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.3612, 0.0258, 0.0, 0.0, 0.3182, -0.4939, -0.4404, -0.2263, 0.0, -0.5719, -0.1531, -0.5106, 0.0, 0.0, 0.0, 0.0, 0.0, -0.4939, -0.5267, -0.6249, -0.5994, 0.5106, -0.7096, -0.128, 0.0, 0.0, -0.296, 0.0, -0.128]



```python

tweets = api.user_timeline("@BBCNews", page=1) 





```


```python
type(tweets)
tweet = tweets[0]
type(tweet)
```




    tweepy.models.Status




```python
tweet.text
```




    'RT @SallyBundockBBC: Apparently it’s #bluemonday2018 - “the most depressing day of the year”. I brought in cup cakes and banana bread for t…'




```python
compound = analyzer.polarity_scores(tweet.text)["compound"]
compound
```




    -0.4391




```python
#     # Variables for holding sentiments
#     compound_list = []

    
#     # Loop through 10 pages of tweets (total 200 tweets)
#     for page in twee(api.user_timeline, id=user).pages(20):

   

        public_tweets = api.search(rpp=100)
        page = page[0]
        tweet = json.dumps(page._json, indent=3)
        tweet = json.loads(tweet)
        text = tweet['text']

        # Run Vader Analysis on each tweet
        compound = analyzer.polarity_scores(text)["compound"]
        
        # Add each value to the appropriate array
        compound_list.append(compound)
        
```


```python

        for pages in tweepy.Cursor(api.user_timeline, id=news).items(100):
            page = page[0]
            tweet = json.dumps(page._json, indent=3)
            tweet = json.loads(tweet)
            text = tweet['text']           
        
```


      File "<ipython-input-5-e4f124f8424e>", line 2
        timestamp = []
                ^
    IndentationError: expected an indented block




```python
        sentiments_df = sentiments_df.append(pd.DataFrame({"Date": tweet["created_at"], 
                           "Compound": compound,
                           "Positive": pos,
                           "Negative": neu,
                           "Neutral": neg,
                           "Tweets Ago": counter}, index=[0]))
        
        # Add to counter 
        counter = counter + 1
```


```python
 # Convert sentiments to DataFrame
sentiments_pd = pd.DataFrame.from_dict(sentiments)
sentiments_pd.head()
```


```python
# Create plot
plt.plot(np.arange(len(sentiments_pd["Compound"])),
         sentiments_pd["Compound"], marker="o", linewidth=0.5,
         alpha=0.8)

# # Incorporate the other graph properties
plt.title("Sentiment Analysis of Tweets (%s) for %s" % (time.strftime("%x"), target_user))
plt.ylabel("Tweet Polarity")
plt.xlabel("Tweets Ago")
plt.show()
```
