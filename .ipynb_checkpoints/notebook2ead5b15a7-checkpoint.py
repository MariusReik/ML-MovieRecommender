{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "_cell_guid": "b1076dfc-b9ad-4769-8c92-a6c4dae69d19",
    "_uuid": "8f2839f25d086af736a60e9eeb907d3b93b6e0e5",
    "execution": {
     "iopub.execute_input": "2024-11-14T16:37:11.305859Z",
     "iopub.status.busy": "2024-11-14T16:37:11.305305Z",
     "iopub.status.idle": "2024-11-14T16:37:11.314664Z",
     "shell.execute_reply": "2024-11-14T16:37:11.312955Z",
     "shell.execute_reply.started": "2024-11-14T16:37:11.305812Z"
    }
   },
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.cluster import KMeans\n",
    "from sklearn.metrics.pairwise import cosine_similarity\n",
    "from scipy.sparse import csr_matrix"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Dataset from https://grouplens.org/datasets/movielens/ 20M"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-11-14T16:37:11.318496Z",
     "iopub.status.busy": "2024-11-14T16:37:11.317935Z",
     "iopub.status.idle": "2024-11-14T16:37:34.402575Z",
     "shell.execute_reply": "2024-11-14T16:37:34.401192Z",
     "shell.execute_reply.started": "2024-11-14T16:37:11.318433Z"
    }
   },
   "outputs": [],
   "source": [
    "# Load movies, ratings, and tags data\n",
    "movies = pd.read_csv('/kaggle/input/movielens-20m-dataset/movie.csv')\n",
    "ratings = pd.read_csv('/kaggle/input/movielens-20m-dataset/rating.csv')\n",
    "tags = pd.read_csv('/kaggle/input/movielens-20m-dataset/tag.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-11-14T16:37:34.406488Z",
     "iopub.status.busy": "2024-11-14T16:37:34.405644Z",
     "iopub.status.idle": "2024-11-14T16:37:36.508864Z",
     "shell.execute_reply": "2024-11-14T16:37:36.507131Z",
     "shell.execute_reply.started": "2024-11-14T16:37:34.406401Z"
    }
   },
   "outputs": [],
   "source": [
    "# Data Preprocessing\n",
    "# Calculate average rating and number of ratings for each movie\n",
    "average_ratings = ratings.groupby('movieId')['rating'].mean().reset_index()\n",
    "average_ratings.rename(columns={'rating': 'average_rating'}, inplace=True)\n",
    "num_ratings = ratings.groupby('movieId')['rating'].count().reset_index()\n",
    "num_ratings.rename(columns={'rating': 'num_ratings'}, inplace=True)\n",
    "\n",
    "# Merge average ratings and number of ratings with movies data\n",
    "movies = pd.merge(movies, average_ratings, on='movieId', how='left')\n",
    "movies = pd.merge(movies, num_ratings, on='movieId', how='left')\n",
    "movies['average_rating'] = movies['average_rating'].fillna(movies['average_rating'].mean())\n",
    "movies['num_ratings'] = movies['num_ratings'].fillna(0)\n",
    "\n",
    "# Preprocess genres to make them a single string per movie\n",
    "movies['genres'] = movies['genres'].str.split('|')\n",
    "movies['genres'] = movies['genres'].apply(lambda x: ' '.join(x))\n",
    "\n",
    "# Merge tags with movies to create a more content-rich dataset\n",
    "tags['tag'] = tags['tag'].fillna('')\n",
    "movies_tags = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()\n",
    "movies = pd.merge(movies, movies_tags, on='movieId', how='left')\n",
    "movies['tag'] = movies['tag'].fillna('')\n",
    "\n",
    "# Combine genres and tags into a single content feature\n",
    "movies['content'] = movies['genres'] + ' ' + movies['tag']\n",
    "movies['content'] = movies['content'].fillna('')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-11-14T16:37:36.511352Z",
     "iopub.status.busy": "2024-11-14T16:37:36.510898Z",
     "iopub.status.idle": "2024-11-14T16:37:37.840964Z",
     "shell.execute_reply": "2024-11-14T16:37:37.839564Z",
     "shell.execute_reply.started": "2024-11-14T16:37:36.511309Z"
    }
   },
   "outputs": [],
   "source": [
    "# Create TF-IDF matrix \n",
    "tfidf = TfidfVectorizer(stop_words='english')\n",
    "tfidf_matrix = tfidf.fit_transform(movies['content'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-11-14T16:37:37.844202Z",
     "iopub.status.busy": "2024-11-14T16:37:37.843759Z",
     "iopub.status.idle": "2024-11-14T16:37:44.965130Z",
     "shell.execute_reply": "2024-11-14T16:37:44.964050Z",
     "shell.execute_reply.started": "2024-11-14T16:37:37.844158Z"
    }
   },
   "outputs": [],
   "source": [
    "# Clustering with K-Means\n",
    "num_clusters = 20 \n",
    "kmeans = KMeans(n_clusters=num_clusters, random_state=42)\n",
    "movies['cluster'] = kmeans.fit_predict(tfidf_matrix)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-11-14T16:37:44.968276Z",
     "iopub.status.busy": "2024-11-14T16:37:44.967381Z",
     "iopub.status.idle": "2024-11-14T16:37:45.555426Z",
     "shell.execute_reply": "2024-11-14T16:37:45.553954Z",
     "shell.execute_reply.started": "2024-11-14T16:37:44.968221Z"
    }
   },
   "outputs": [],
   "source": [
    "# Visualization: Frequency of Movie Genres\n",
    "plt.figure(figsize=(14, 8))\n",
    "genre_list = movies['genres'].str.split().explode()\n",
    "sns.countplot(y=genre_list, order=genre_list.value_counts().index)\n",
    "plt.title(\"Frequency of Movie Genres\")\n",
    "plt.xlabel(\"Count\")\n",
    "plt.ylabel(\"Genre\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-11-14T16:37:45.557595Z",
     "iopub.status.busy": "2024-11-14T16:37:45.557148Z",
     "iopub.status.idle": "2024-11-14T16:37:46.147293Z",
     "shell.execute_reply": "2024-11-14T16:37:46.145859Z",
     "shell.execute_reply.started": "2024-11-14T16:37:45.557549Z"
    }
   },
   "outputs": [],
   "source": [
    "# Visualization: Distribution of Average Movie Ratings\n",
    "plt.figure(figsize=(14, 8))\n",
    "sns.histplot(movies['average_rating'], bins=30, kde=True)\n",
    "plt.title(\"Distribution of Average Movie Ratings\")\n",
    "plt.xlabel(\"Average Rating\")\n",
    "plt.ylabel(\"Frequency\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-11-14T16:37:46.149380Z",
     "iopub.status.busy": "2024-11-14T16:37:46.148949Z",
     "iopub.status.idle": "2024-11-14T16:37:46.168476Z",
     "shell.execute_reply": "2024-11-14T16:37:46.167034Z",
     "shell.execute_reply.started": "2024-11-14T16:37:46.149336Z"
    }
   },
   "outputs": [],
   "source": [
    "def recommend_movies(input_movies, input_genres, movies_df, tfidf_matrix, num_recommendations=5, rating_weight=0.1, similarity_weight=0.8, popularity_weight=0.1, min_rating_threshold=3.0):\n",
    "    \"\"\"\n",
    "    Recommend movies based on genres, clustering, and content similarity.\n",
    "    \n",
    "    Parameters:\n",
    "    input_movies (list): List of movies the user likes.\n",
    "    input_genres (list): List of genres the user likes.\n",
    "    movies_df (DataFrame): DataFrame containing movie information.\n",
    "    tfidf_matrix (sparse matrix): TF-IDF representation of movie content.\n",
    "    num_recommendations (int): Number of movies to recommend.\n",
    "    rating_weight (float): Weight for the average rating in the final score.\n",
    "    similarity_weight (float): Weight for the similarity score in the final score.\n",
    "    popularity_weight (float): Weight for the number of ratings in the final score.\n",
    "    min_rating_threshold (float): Minimum average rating for recommended movies.\n",
    "    \n",
    "    Returns:\n",
    "    DataFrame: DataFrame containing recommended movies and their details.\n",
    "    \"\"\"\n",
    "    \n",
    "    input_clusters = movies_df[movies_df['title'].isin(input_movies)]['cluster'].unique()\n",
    "\n",
    "    cluster_filtered = movies_df[movies_df['cluster'].isin(input_clusters)]\n",
    "\n",
    "    if cluster_filtered.empty:\n",
    "        cluster_filtered = movies_df\n",
    "\n",
    "    cluster_filtered = cluster_filtered[cluster_filtered['average_rating'] >= min_rating_threshold]\n",
    "\n",
    "    input_content = ' '.join(input_genres)\n",
    "    for movie in input_movies:\n",
    "        if movie in movies_df['title'].values:\n",
    "            input_content += ' ' + movies_df[movies_df['title'] == movie]['content'].values[0]\n",
    "\n",
    "    input_vector = tfidf.transform([input_content])\n",
    "\n",
    "    cosine_sim = cosine_similarity(input_vector, tfidf_matrix[cluster_filtered.index]).flatten()\n",
    "\n",
    "    cluster_filtered = cluster_filtered.copy()\n",
    "    cluster_filtered['similarity'] = cosine_sim\n",
    "\n",
    "    cluster_filtered['normalized_similarity'] = (cluster_filtered['similarity'] - cluster_filtered['similarity'].min()) / \\\n",
    "                                                (cluster_filtered['similarity'].max() - cluster_filtered['similarity'].min())\n",
    "\n",
    "    cluster_filtered['normalized_num_ratings'] = (cluster_filtered['num_ratings'] - cluster_filtered['num_ratings'].min()) / \\\n",
    "                                                 (cluster_filtered['num_ratings'].max() - cluster_filtered['num_ratings'].min())\n",
    "\n",
    "    median_num_ratings = cluster_filtered['num_ratings'].median()\n",
    "    cluster_filtered['rating_penalty'] = cluster_filtered['num_ratings'].apply(lambda x: 0.8 if x < median_num_ratings else 1.0)\n",
    "\n",
    "    cluster_filtered['final_score'] = ((rating_weight * cluster_filtered['average_rating']) + \\\n",
    "                                       (similarity_weight * cluster_filtered['normalized_similarity']) + \\\n",
    "                                       (popularity_weight * cluster_filtered['normalized_num_ratings'])) * cluster_filtered['rating_penalty']\n",
    "\n",
    "    recommendations = cluster_filtered.sort_values(by='final_score', ascending=False).head(num_recommendations)\n",
    "\n",
    "    recommendations['average_rating'] = recommendations['average_rating'].round(2)\n",
    "    recommendations['similarity'] = recommendations['similarity'].round(2)\n",
    "    recommendations.reset_index(drop=True, inplace=True)\n",
    "    recommendations.index += 1\n",
    "\n",
    "    return recommendations[['title', 'genres', 'average_rating', 'similarity', 'final_score']].rename(columns={\n",
    "        'title': 'Movie Title',\n",
    "        'genres': 'Genres',\n",
    "        'average_rating': 'Average Rating',\n",
    "        'similarity': 'Similarity Score',\n",
    "        'final_score': 'Final Score'\n",
    "    })\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-11-14T16:37:46.170772Z",
     "iopub.status.busy": "2024-11-14T16:37:46.170260Z",
     "iopub.status.idle": "2024-11-14T16:37:46.251536Z",
     "shell.execute_reply": "2024-11-14T16:37:46.250181Z",
     "shell.execute_reply.started": "2024-11-14T16:37:46.170680Z"
    }
   },
   "outputs": [],
   "source": [
    "input_movies = ['The Matrix']\n",
    "input_genres = ['Action', 'Sci-Fi']\n",
    "recommended_movies = recommend_movies(input_movies, input_genres, movies, tfidf_matrix)\n",
    "print(recommended_movies)"
   ]
  }
 ],
 "metadata": {
  "kaggle": {
   "accelerator": "none",
   "dataSources": [
    {
     "datasetId": 339,
     "sourceId": 77759,
     "sourceType": "datasetVersion"
    }
   ],
   "dockerImageVersionId": 30786,
   "isGpuEnabled": false,
   "isInternetEnabled": false,
   "language": "python",
   "sourceType": "notebook"
  },
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.11"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
