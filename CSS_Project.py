#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 03:23:50 2024

@author: rowynnaidoo
"""

import pandas as pd

file = pd.read_csv("movie_dataset.csv")

print(file)

print(file.describe())

file.dropna(inplace = True)
df = file.reset_index(drop=True)
print(len(df))

highest_rank = df['Rank'].max()
print(highest_rank)
highest_title = df.loc[len(df)-1, 'Title']
print("Lowest ranked movie is:", highest_title)

print("Highest ranked movie is:", df.loc[0, 'Title'])

highest_rated = df['Rating'].max()
print("Highest rated movie is:", df.loc[df['Rating'].idxmax(), 'Title'])

print("The average revenue is:", df['Revenue (Millions)'].mean())

filtered_df = df[(df['Year'] >= 2015) & (df['Year'] <= 2017)]
print("The average revenue between 2015 and 2017 is:", filtered_df['Revenue (Millions)'].mean())

print(df['Year'].value_counts())

director = "Christopher Nolan"
print("The number of movies directed by Christopher Nolan were:", (df['Director'] == director).sum())

print("The number of movies in the dataset that have a rating of at least 8.0 were:", (df['Rating'] >= 8).sum())

filtered_df = df[(df['Director'] == "Christopher Nolan")]
print("The median of the ratings of movies directed by Christopher Nolan were:", filtered_df['Rating'].median())

print("Average ratings of all years",df.groupby('Year')['Rating'].mean())




from collections import Counter

all_actors = ','.join(df['Actors']).split(', ')


actor_counts = Counter(all_actors)

most_common_actor = actor_counts.most_common(5)

print(f"The actor that appears the most is: {most_common_actor}")


movies_2006 = df[df['Year'] == 2006].shape[0]
movies_2016 = df[df['Year'] == 2016].shape[0]

percentage_increase = ((movies_2016 - movies_2006) / movies_2006) * 100

print("The percenage increase in movies made between 2006 and 2016 is", percentage_increase)


df['unique_genres'] = df['Genre'].apply(lambda x: set(x.split(', ')))

# Combine all unique genres into a single list
all_unique_genres = [genre for genres_set in df['unique_genres'] for genre in genres_set]

# Count the occurrences of each unique genre
unique_genre_counts = pd.Series(all_unique_genres).value_counts()

print("Number of different types of genres:")
print(len(unique_genre_counts))









import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


director_revenue = df.groupby('Director')['Revenue (Millions)'].sum().reset_index()

director_list = director_revenue['Director'].tolist()
revenue_list = director_revenue['Revenue (Millions)'].tolist()
print(director_list)
print(director_revenue)

# Fit a linear regression model
model = np.polyfit(director_list, revenue_list, 1)
predict = np.poly1d(model)

# Calculate R-squared
r2 = r2_score(director_list, predict(director_revenue))
print("R-squared:", r2)

# Scatter plot
plt.scatter(director_list, director_revenue, label='Actual data')

# Regression line plot
plt.plot(director_list, predict(director_revenue), color='red', label='Regression line')

# Labels and title
plt.xlabel('Director')
plt.ylabel('Revenue')
plt.title('Scatter Plot with Regression Line')

# Show legend
plt.legend()

# Display the plot
plt.show()


counter_data = Counter({'apple': 3, 'banana': 2, 'orange': 1})

# Separate Counter data into two lists: keys and values
keys_list = list(counter_data.keys())
values_list = list(counter_data.values())

print("List of keys:")
print(keys_list)

print("List of values:")
print(values_list)


























