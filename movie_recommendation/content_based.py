import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#read the csv file 
df = pd.read_csv("movie_recommendation/processed_movie_data.csv")

#Apply TD-IDF Vectorization
tfidf = TfidfVectorizer(stop_words = 'english')
tfidf_matrix = tfidf.fit_transform(df['combined_features'])
'''
TfidfVectorizer() creates our TF-IDF converter.
stop_words="english" removes common English words such as "the", "is", "and".
fit_transform() learns the vocabulary and converts movie descriptions into numerical vectors.
'''
print("TF-IDF Matrix shape:", tfidf_matrix.shape)

#calculate cosine similarity
#For more understanding about cosine similarity, please refer to the following link: https://www.geeksforgeeks.org/dbms/cosine-similarity/
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
print("Cosine similarity created successfully!")
print("Cosine similarity shape:", cosine_sim.shape)

#crete movie title index 
indices = pd.Series(
    df.index, 
    index=df["title"].str.lower().str.strip()
)

#Recommendation function
def recommend_movies(title, num_recommendation =5):
    
    title = title.lower()

    if title not in indices:
        return "Movie not found in the dataset. Please check the title and try again."
    
    idx = indices[title]
    if isinstance(idx, pd.Series):
        idx = idx.iloc[0]
    #Get similarity scores for all movies with the given movie
    similarity_scores = list(enumerate(cosine_sim[idx]))

    #Sort movies by similarity
    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    #Get top similar movies (excluding itself)
    similarity_scores = similarity_scores[1: num_recommendation + 1]

    #Extract movies indices
    movie_indices = [i[0] for i in similarity_scores]

    return df["title"].iloc[movie_indices]

#test recommendation system
movie_name = input("Enter a movie name: ")
recommendations = recommend_movies(movie_name)

print("\nRecommended movies based on your input:")
print(recommendations)

