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