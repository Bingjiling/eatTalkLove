import pandas as pd 

df = pd.read_csv('./data/review.csv')
df = df.ix[range(10000),:]
df = df[['text', 'stars']]

df = df.dropna() 

df.to_pickle('./data/review.pkl')