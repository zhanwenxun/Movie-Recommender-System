# tagcloud.py
# Generate current user's personalized Tag Cloud

import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt

genre_list = ["Animation", "Children", "Comedy", "Adventure", "Fantasy",
        "Romance", "Drama", "Action", "Crime", "Thriller", "Horror", "SciFi",
        "Documentary", "War", "Musical", "Mystery", "FilmNoir", "Western"]

# Get all genres of movies with user rating greater than 3
def getText(user_id):
    df = pd.read_csv('new_u.data', sep=',', names=['user', 'item', 'rating', 'timestamp'])
    # acquire current user rated movies (rating >= 3)
    rated_movies = df[df['user'] == user_id]
    high_rated = rated_movies[rated_movies['rating'] >= 3]['item'].values.tolist()
    print(len(high_rated))
    
    movies = pd.read_excel("movies.xlsx")

    text = ''
    for iid in high_rated:
        # acquire genre array of this movie
        movies2 = movies[movies['movie_id'] == iid]
        movies2 = movies2[genre_list]
        genres = movies2.apply(lambda row: row[row == 1].index.values, axis=1).tolist()
        result = ' '.join(genres[0])
        text = text + result + ' '
    return text

# Calculate the frequency of each word in the text
def getFrequencyDictForText(sentence):
    tmpDict = {}
    
    # making dict for counting frequencies
    for text in sentence.split(" "):
        val = tmpDict.get(text, 0)
        print(val)
        tmpDict[text] = val + 1
    print(tmpDict)
    
    return tmpDict

# Plot from the calculated word frequency
def makeImage(text):
    wc = WordCloud(background_color="white")
    # generate word cloud
    wc.generate_from_frequencies(text)
    
    # show
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.savefig('tagcloud.jpg')

# main.py is called like this
# text = getText(user_id)
# makeImage(getFrequencyDictForText(text))