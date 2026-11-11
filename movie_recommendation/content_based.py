import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("movie_recommendation/processed_movie_data.csv")

# Handle missing values in text columns
feature_columns = {
    "genre": "genre",
    "description": "description",
    "director": "director",
    "actors": "actors"
}

for column in feature_columns.values():
    df[column] = df[column].fillna("").astype(str)


# ==========================================
# 2. CREATE TF-IDF MATRICES
# ==========================================

tfidf_matrices = {}
similarity_matrices = {}

for feature, column in feature_columns.items():

    tfidf = TfidfVectorizer(stop_words="english")

    tfidf_matrix = tfidf.fit_transform(df[column])

    tfidf_matrices[feature] = tfidf_matrix

    similarity_matrices[feature] = cosine_similarity(
        tfidf_matrix,
        tfidf_matrix
    )

    print(f"{feature.capitalize()} similarity matrix created.")

    print(
        f"Matrix shape: "
        f"{similarity_matrices[feature].shape}"
    )


# ==========================================
# 3. FEATURE WEIGHTS
# ==========================================

weights = {
    "description": 0.70,
    "director": 0.15,
    "actors": 0.15
}


# ==========================================
# 4. CREATE MOVIE TITLE INDEX
# ==========================================

df["normalized_title"] = (
    df["title"]
    .astype(str)
    .str.lower()
    .str.strip()
)

indices = pd.Series(
    df.index,
    index=df["normalized_title"]
)


# ==========================================
# 5. RECOMMENDATION FUNCTION
# ==========================================

def recommend_movies(title, num_recommendations=5):

    title = title.lower().strip()

    if title not in indices:
        return "Movie not found in the dataset. Please check the title."

    idx = indices[title]

    # Handle duplicate movie titles
    if isinstance(idx, pd.Series):
        idx = idx.iloc[0]

    # Calculate weighted similarity
    final_scores = sum(
        weights[feature] * similarity_matrices[feature][idx]
        for feature in weights
    )

    # Sort movies by similarity score
    similarity_scores = list(enumerate(final_scores))

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    # Exclude the input movie itself
    similarity_scores = [
        item for item in similarity_scores
        if item[0] != idx
    ]

    # Select Top-N recommendations
    top_movies = similarity_scores[:num_recommendations]

    # Extract movie information
    movie_indices = [item[0] for item in top_movies]

    recommendations = df.iloc[movie_indices][
        ["title", "genre", "rating", "year"]
    ].copy()

    recommendations["similarity_score"] = [
        round(item[1], 4) for item in top_movies
    ]

    return recommendations.reset_index(drop=True)


# ==========================================
# 6. TEST THE RECOMMENDATION SYSTEM
# ==========================================

if __name__ == "__main__":

    print("\nIMDb Movie Recommendation System")
    print("--------------------------------")

    movie_name = input("Enter a movie name: ")

    recommendations = recommend_movies(movie_name)

    print("\nRecommended Movies:")

    if isinstance(recommendations, str):
        print(recommendations)

    elif recommendations.empty:
        print("No recommendations found.")

    else:
        print(recommendations.to_string(index=False))
