#!/usr/bin/env python
# coding: utf-8

# export the model
# genres
# fix dataset and export it
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import joblib
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv("BUILD_WEEK_3/build_week_3/Spotify_data.csv")

df.head(2)

# create year column from release date 
df['year'] = pd.DatetimeIndex(df['release_date']).year

# drop categorical columns
df.drop(columns=['release_date','id','id_artists'],inplace = True)

# remove duplicates
print("Origional:", df.shape)
df = df.drop_duplicates()
print("Cleaned:", df.shape)

# label encode artists so It can select the same artist
enc = LabelEncoder()
df['artists'] = enc.fit_transform(df['artists'])

# normalize data
for i in df.columns[1:]:
    df[i] = df[i]/df[i].max()

X = df.drop(columns=['name']).values


neighbors = 20
nn = NearestNeighbors(n_neighbors=neighbors + 1)
nn.fit(X)


# returns the nearest neighbor number of nearest songs given a song object
def song_suggester(song_obj):
    distance, neighbors = nn.kneighbors(np.array([song_obj]))
    suggestions = []
    for i in neighbors[0][1:]:
        suggestions.append([df['name'].iloc[i],df['artists'].iloc[i]*21594])
    sug = pd.DataFrame(suggestions,columns = ['song','artist'])
    sug['artist'] = enc.inverse_transform(sug['artist'].to_numpy().astype(int))
    return sug
# returns a song from song suggester from an exact name
def song(name):
    song = 'song'
    try:
        song = song_suggester(X[df['name']==name][0])
    except:
        print("Invalid Song Name")
    return song

def get_x():
    return X

def get_list(sug):
    sug["combined"] = sug["song"] + sug["artist"]
    return sug["combined"].to_list()


# Just call song(song_name) to get a dataframe with 20 suggestions [title,artist]

# example for creating similar songs
example = 92123

print('Song:',df['name'].iloc[example])

suggestions = song_suggester(X[example])

suggestions.head()


df.head()

df.to_csv('data.csv',index=False)

joblib.dump(nn,'model.z')
joblib.dump(enc,'encoder.z')

