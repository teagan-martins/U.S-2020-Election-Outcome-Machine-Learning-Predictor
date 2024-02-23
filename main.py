import numpy as np
import pandas as pd
from textblob import TextBlob
import plotly.graph_objects as go

def pol(tweet):
    if isinstance(tweet, str):  # Check if tweet is a string
        return TextBlob(tweet).sentiment.polarity
    else:
        return 0  # Return a default value for NaN or non-string values

trump = pd.read_csv('hashtag_donaldtrump.csv')
biden = pd.read_csv('hashtag_joebiden.csv')

trump["Polarity"] = trump["tweet"].apply(pol)

biden["Polarity"]=biden["tweet"].apply(pol)

trump["sentiment"]=np.where(trump["Polarity"]>0,"Positive","Negative")
trump["sentiment"][trump["Polarity"]==0]="Neutral"

biden["sentiment"]=np.where(biden["Polarity"]>0,"Positive","Negative")
biden["sentiment"][biden["Polarity"]==0]="Neutral"

trump_neutral=trump[trump["Polarity"]==0]

biden_neutral=biden[biden["Polarity"]==0]

trump.drop(trump[trump["Polarity"]==0].index, inplace= True)

biden.drop(biden[biden["Polarity"]==0].index, inplace= True)

np.random.seed(10)
no=293
drop_index=np.random.choice(trump.index, no, replace=False)
trump=trump.drop(drop_index)

np.random.seed(10)
no=293
drop_index=np.random.choice(biden.index, no, replace=False)
biden=biden.drop(drop_index)

count_trump=trump.groupby('sentiment').count()

count_biden=biden.groupby('sentiment').count()

name=["Trump","Biden"]
list_pos=[count_trump['Polarity'][1],count_biden['Polarity'][1]]
list_neg=[count_trump['Polarity'][0],count_biden['Polarity'][0]]

fig=go.Figure(data=[
    go.Bar(name='+ve',x=name, y=list_pos),
    go.Bar(name='-ve', x=name, y=list_neg)
])

fig.update_layout(barmode='group')
fig.show()

ratio_trump = list_pos[0] / (list_pos[0] + list_neg[0])
ratio_biden = list_pos[1] / (list_pos[1] + list_neg[1])

if ratio_trump > ratio_biden:
    print("Prediction: Tweets about Donald Trump are typically more positive, suggesting Donald Trump is favoured in the election.")
    print(f"% of Tweets about Donald Trump that are positive: {ratio_trump}")
    print(f"% of Tweets about Joe Biden that are positive: {ratio_biden}")
elif ratio_biden > ratio_trump:
    print("Prediction: Tweets about Joe Biden are typically more positive, suggesting Joe Biden is favoured in the election.")
    print(f"% of Tweets about Donald Trump that are positive: {ratio_trump}")
    print(f"% of Tweets about Joe Biden that are positive: {ratio_biden}")
else:
    print("Prediction: Both candidates have similar positive tweet ratios.")
    print(f"% of Tweets about Donald Trump that are positive: {ratio_trump}")
    print(f"% of Tweets about Joe Biden that are positive: {ratio_biden}")

positive_trump_tweets = trump[trump['sentiment'] == 'Positive']

positive_trump_tweets['likes'] = pd.to_numeric(positive_trump_tweets['likes'], errors='coerce')

# Sum up the likes on positive tweets
trump_positive_likes = positive_trump_tweets['likes'].sum()

negative_trump_tweets = trump[trump['sentiment'] == 'Negative']

negative_trump_tweets['likes'] = pd.to_numeric(negative_trump_tweets['likes'], errors='coerce')

# Sum up the likes on positive tweets
trump_negative_likes = negative_trump_tweets['likes'].sum()

positive_biden_tweets = biden[biden['sentiment'] == 'Positive']

positive_biden_tweets['likes'] = pd.to_numeric(positive_biden_tweets['likes'], errors='coerce')

# Sum up the likes on positive tweets
biden_positive_likes = positive_biden_tweets['likes'].sum()

negative_biden_tweets = biden[biden['sentiment'] == 'Negative']

negative_biden_tweets['likes'] = pd.to_numeric(negative_biden_tweets['likes'], errors='coerce')

# Sum up the likes on positive tweets
biden_negative_likes = negative_biden_tweets['likes'].sum()

ratio_likes_trump = trump_positive_likes / (trump_positive_likes + trump_negative_likes)
ratio_likes_biden = biden_positive_likes / (biden_positive_likes + biden_negative_likes)

if ratio_likes_trump > ratio_likes_biden:
    print("Prediction: Likes regarding Donald Trump are typically on positive tweets about him, suggesting Donald Trump is favoured in the election.")
    print(f"% of likes on tweets about Donald Trump that are positive: {ratio_likes_trump}")
    print(f"% of likes on tweets about Joe Biden that are positive: {ratio_likes_biden}")
elif ratio_likes_biden > ratio_likes_trump:
    print("Prediction: Likes regarding Joe Biden are typically on positive tweets about him, suggesting Joe Biden is favoured in the election.")
    print(f"% of likes on tweets about Donald Trump that are positive: {ratio_likes_trump}")
    print(f"% of likes on tweets about Joe Biden that are positive: {ratio_likes_biden}")
else:
    print("Prediction: Both candidates have similar positive tweet like ratios.")
    print(f"% of likes on tweets about Donald Trump that are positive: {ratio_likes_trump}")
    print(f"% of likes on tweets about Joe Biden that are positive: {ratio_likes_biden}")

positive_trump_tweets = trump[trump['sentiment'] == 'Positive']

positive_trump_tweets['retweet_count'] = pd.to_numeric(positive_trump_tweets['retweet_count'], errors='coerce')

# Sum up the likes on positive tweets
trump_positive_retweets = positive_trump_tweets['retweet_count'].sum()

negative_trump_tweets = trump[trump['sentiment'] == 'Negative']

negative_trump_tweets['retweet_count'] = pd.to_numeric(negative_trump_tweets['retweet_count'], errors='coerce')

# Sum up the likes on positive tweets
trump_negative_retweets = negative_trump_tweets['retweet_count'].sum()

positive_biden_tweets = biden[biden['sentiment'] == 'Positive']

positive_biden_tweets['retweet_count'] = pd.to_numeric(positive_biden_tweets['retweet_count'], errors='coerce')

# Sum up the likes on positive tweets
biden_positive_retweets = positive_biden_tweets['retweet_count'].sum()

negative_biden_tweets = biden[biden['sentiment'] == 'Negative']

negative_biden_tweets['retweet_count'] = pd.to_numeric(negative_biden_tweets['retweet_count'], errors='coerce')

# Sum up the likes on positive tweets
biden_negative_retweets = negative_biden_tweets['retweet_count'].sum()

ratio_retweets_trump = trump_positive_retweets / (trump_positive_retweets + trump_negative_retweets)
ratio_retweets_biden = biden_positive_retweets / (biden_positive_retweets + biden_negative_retweets)

if ratio_retweets_trump > ratio_retweets_biden:
    print("Prediction: Retweets regarding Donald Trump are typically on positive tweets about him, suggesting Donald Trump is favoured in the election.")
    print(f"% of retweets on tweets about Donald Trump that are positive: {ratio_retweets_trump}")
    print(f"% of retweets on tweets about Joe Biden that are positive: {ratio_retweets_biden}")
elif ratio_retweets_biden > ratio_retweets_trump:
    print("Prediction: Retweets regarding Joe Biden are typically on positive tweets about him, suggesting Joe Biden is favoured in the election.")
    print(f"% of retweets on tweets about Donald Trump that are positive: {ratio_retweets_trump}")
    print(f"% of retweets on tweets about Joe Biden that are positive: {ratio_retweets_biden}")
else:
    print("Prediction: Both candidates have similar positive retweet ratios.")
    print(f"% of retweets on tweets about Donald Trump that are positive: {ratio_retweets_trump}")
    print(f"% of retweets on tweets about Joe Biden that are positive: {ratio_retweets_biden}")
