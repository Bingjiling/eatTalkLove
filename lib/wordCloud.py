import pandas as pd 
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

df = pd.read_pickle('./data/review.pkl')
goodDf = df[(df.stars == 5)]
badDf = df[(df.stars == 1)]
goodReviews = goodDf.text
badReviews = badDf.text


text ='' 
for review in goodReviews:
	text = text+' '+review

wordcloud = WordCloud(stopwords=STOPWORDS).generate(text)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

text ='' 
for review in badReviews:
	text = text+' '+review

wordcloud = WordCloud(stopwords=STOPWORDS).generate(text)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()