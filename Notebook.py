import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import gradio as gr

# Load movies, ratings, and tags data
movies = pd.read_csv(r'C:\Users\mariu\Documents\DAT158\ML2\movielens-20m\movie.csv')
ratings = pd.read_csv(r'C:\Users\mariu\Documents\DAT158\ML2\movielens-20m\rating.csv')
tags = pd.read_csv(r'C:\Users\mariu\Documents\DAT158\ML2\movielens-20m\tag.csv')


# Data Preprocessing
# Calculate average rating and number of ratings for each movie
average_ratings = ratings.groupby('movieId')['rating'].mean().reset_index()
average_ratings.rename(columns={'rating': 'average_rating'}, inplace=True)
num_ratings = ratings.groupby('movieId')['rating'].count().reset_index()
num_ratings.rename(columns={'rating': 'num_ratings'}, inplace=True)

# Merge average ratings and number of ratings with movies data
movies = pd.merge(movies, average_ratings, on='movieId', how='left')
movies = pd.merge(movies, num_ratings, on='movieId', how='left')
movies['average_rating'] = movies['average_rating'].fillna(movies['average_rating'].mean())
movies['num_ratings'] = movies['num_ratings'].fillna(0)

# Preprocess genres to make them a single string per movie
movies['genres'] = movies['genres'].str.split('|')
movies['genres'] = movies['genres'].apply(lambda x: ' '.join(x))

# Merge tags with movies to create a more content-rich dataset
tags['tag'] = tags['tag'].fillna('')
movies_tags = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
movies = pd.merge(movies, movies_tags, on='movieId', how='left')
movies['tag'] = movies['tag'].fillna('')

# Combine genres and tags into a single content feature
movies['content'] = movies['genres'] + ' ' + movies['tag']
movies['content'] = movies['content'].fillna('')

# Create TF-IDF matrix 
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['content'])

# Clustering with K-Means
num_clusters = 20 
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
movies['cluster'] = kmeans.fit_predict(tfidf_matrix)

# Visualization: Frequency of Movie Genres
plt.figure(figsize=(14, 8))
genre_list = movies['genres'].str.split().explode()
sns.countplot(y=genre_list, order=genre_list.value_counts().index)
plt.title("Frequency of Movie Genres")
plt.xlabel("Count")
plt.ylabel("Genre")
plt.show()

# Visualization: Distribution of Average Movie Ratings
plt.figure(figsize=(14, 8))
sns.histplot(movies['average_rating'], bins=30, kde=True)
plt.title("Distribution of Average Movie Ratings")
plt.xlabel("Average Rating")
plt.ylabel("Frequency")
plt.show()

# Recommendation function
def recommend_movies(input_movies, input_genres, movies_df, tfidf_matrix, num_recommendations=5, rating_weight=0.1, similarity_weight=0.8, popularity_weight=0.1, min_rating_threshold=3.0):
    input_clusters = movies_df[movies_df['title'].isin(input_movies)]['cluster'].unique()
    cluster_filtered = movies_df[movies_df['cluster'].isin(input_clusters)]

    if cluster_filtered.empty:
        cluster_filtered = movies_df

    cluster_filtered = cluster_filtered[cluster_filtered['average_rating'] >= min_rating_threshold]

    input_content = ' '.join(input_genres)
    for movie in input_movies:
        if movie in movies_df['title'].values:
            input_content += ' ' + movies_df[movies_df['title'] == movie]['content'].values[0]

    input_vector = tfidf.transform([input_content])
    cosine_sim = cosine_similarity(input_vector, tfidf_matrix[cluster_filtered.index]).flatten()

    cluster_filtered = cluster_filtered.copy()
    cluster_filtered['similarity'] = cosine_sim

    cluster_filtered['normalized_similarity'] = (cluster_filtered['similarity'] - cluster_filtered['similarity'].min()) / \
                                                (cluster_filtered['similarity'].max() - cluster_filtered['similarity'].min())

    cluster_filtered['normalized_num_ratings'] = (cluster_filtered['num_ratings'] - cluster_filtered['num_ratings'].min()) / \
                                                 (cluster_filtered['num_ratings'].max() - cluster_filtered['num_ratings'].min())

    median_num_ratings = cluster_filtered['num_ratings'].median()
    cluster_filtered['rating_penalty'] = cluster_filtered['num_ratings'].apply(lambda x: 0.8 if x < median_num_ratings else 1.0)

    cluster_filtered['final_score'] = ((rating_weight * cluster_filtered['average_rating']) + \
                                       (similarity_weight * cluster_filtered['normalized_similarity']) + \
                                       (popularity_weight * cluster_filtered['normalized_num_ratings'])) * cluster_filtered['rating_penalty']

    recommendations = cluster_filtered.sort_values(by='final_score', ascending=False).head(num_recommendations)
    recommendations['average_rating'] = recommendations['average_rating'].round(2)
    recommendations['similarity'] = recommendations['similarity'].round(2)
    recommendations.reset_index(drop=True, inplace=True)
    recommendations.index += 1

    return recommendations[['title', 'genres', 'average_rating', 'similarity', 'final_score']].rename(columns={
        'title': 'Movie Title',
        'genres': 'Genres',
        'average_rating': 'Average Rating',
        'similarity': 'Similarity Score',
        'final_score': 'Final Score'
    })

# Example usage
input_movies = ['The Matrix']
input_genres = ['Action', 'Sci-Fi']
recommended_movies = recommend_movies(input_movies, input_genres, movies, tfidf_matrix)
print(recommended_movies)


def gradio_recommend_movies(input_movies, input_genres):
    recommended_movies = recommend_movies(input_movies.split(','), input_genres.split(','), movies, tfidf_matrix)
    return recommended_movies[['Movie Title', 'Genres']]

# Create the Gradio interface
inputs = [
    gr.Textbox(lines=1, placeholder="Enter movie names separated by commas", label="Movies"),
    gr.Textbox(lines=1, placeholder="Enter genres separated by commas", label="Genres"),
]

outputs = gr.Dataframe(label="Recommended Movies")


interface = gr.Interface(
    fn=gradio_recommend_movies,
    inputs=inputs,
    outputs=outputs,
    title="Movie Recommender",
)
interface.launch()