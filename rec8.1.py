import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import sys
import pickle
X1=input("Enter X1")
######################################
#Load movies data
movies = pd.read_csv('ml-20m/movies.csv')
genome_scores = pd.read_csv('ml-20m/genome-scores.csv')
tags = pd.read_csv('ml-20m/tags.csv')
genome_tags = pd.read_csv('ml-20m/genome-tags.csv')
#Use ratings data to downsample tags data to only movies with ratings 
ratings = pd.read_csv('ml-20m/ratings.csv')
#ratings = ratings.drop_duplicates('movieId')
#######################################
X2=input("Enter X2")
print(movies.tail())
movies['genres'] = movies['genres'].str.replace('|',' ')
print(len(movies.movieId.unique()))
print(len(ratings.movieId.unique()))
########################################
#limit ratings to user ratings that have rated more that 100 movies -- 
#Otherwise it becomes impossible to pivot the rating dataframe later for collaborative filtering.

ratings_f = ratings.groupby('userId').filter(lambda x: len(x) >= 300)   #this line changed

# list the movie titles that survive the filtering
movie_list_rating = ratings_f.movieId.unique().tolist()
#############################################
# no worries: we have kept ?% of the original movie titles in ratings data frame
print(len(ratings_f.movieId.unique())/len(movies.movieId.unique()) * 100)
##############################################
# but only ?% of the users 
print(len(ratings_f.userId.unique())/len(ratings.userId.unique()) * 100)
##############################################
#filter the movies data frame
movies = movies[movies.movieId.isin(movie_list_rating)]
##############################################
print(movies.head(3))
##############################################
X3=input("Enter X3")
# map movie to id:
Mapping_file = dict(zip(movies.title.tolist(), movies.movieId.tolist()))
##############################################
tags.drop(['timestamp'],1, inplace=True)
ratings_f.drop(['timestamp'],1, inplace=True)
##############################################
# create a mixed dataframe of movies title, genres 
# and all user tags given to each movie
mixed = pd.merge(movies, tags, on='movieId', how='left')
print(mixed.head(3))
################################################
X4=input("Enter X4")
# create metadata from tags and genres
mixed.fillna("", inplace=True)
mixed = pd.DataFrame(mixed.groupby('movieId')['tag'].apply(lambda x: "%s" % ' '.join(x)))
Final = pd.merge(movies, mixed, on='movieId', how='left')
Final ['metadata'] = Final[['tag', 'genres']].apply(lambda x: ' '.join(x), axis = 1)
print(Final[['movieId','title','metadata']].head(3))
##################################################
#############################################################
X5=input("Enter X5")
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(Final['metadata'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=Final.index.tolist())
print(tfidf_df.shape)
########################################################
tfidf_df2=pd.DataFrame(tfidf_df.head(10))    #this line changed
print(tfidf_df2)
###########################################
tfidf_df3=tfidf_df.iloc[:30,:300]   #this line changed
print(tfidf_df3)



#########################################################
print("Final.index.tolist():",Final.index.tolist())
#########################################################
X6=input("Enter X6")
# Compress with SVD
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=200)
latent_matrix = svd.fit_transform(tfidf_df3)
# plot var expalined to see what latent dimensions to use
explained = svd.explained_variance_ratio_.cumsum()
plt.plot(explained, '.-', ms = 16, color='red')
plt.xlabel('Singular value components', fontsize= 12)
plt.ylabel('Cumulative percent of variance', fontsize=12)        
plt.show()
tfidf_df=tfidf_df3  #this line changed
###########################################################
########
#######
###
#
############################################################
###########################################################
#number of latent dimensions to keep
n = 30   #this line changed
#latent_matrix_1_df = pd.DataFrame(latent_matrix[:,0:n], index=Final.title.tolist())
latent_matrix_1_df = pd.DataFrame(latent_matrix[:,0:n], index=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30])#this line changed
###########################################################
# our content latent matrix:
print(latent_matrix.shape)
print(ratings_f.head())
############################################################



