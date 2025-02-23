import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys


def load_dataset(filepath="dataset/imdb_top_1000.csv"):
    """
    Load and preprocess the dataset from the given file path.

    The function reads a CSV file containing movie data, selects relevant
    columns, and applies basic text preprocessing (lowercasing and punctuation
    removal) to the "Overview" column.

    :param filepath: str
        Path to the CSV dataset file.

    :return: pd.DataFrame
        A DataFrame containing movie titles and their cleaned text
        descriptions.
    """
    data = pd.read_csv(filepath)
    df = data[['Series_Title', 'Overview']].dropna()

    # Convert text to lowercase
    df['Overview'] = df['Overview'].str.lower()

    # Remove punctuation
    df['Overview'] = df['Overview'].str.replace(r'[^\w\s]', '', regex=True)

    return df


def preprocess_text(query):
    """
    Preprocess the text by converting it to lowercase

    :param query: str
        The input text query to preprocess.

    :return: str
        The lowercase version of the input query.
    """
    return query.lower()  # Convert to lowercase


def build_tfidf_matrix(text):
    """
    Build a TF-IDF vector representation of a given corpus.

    The function applies TF-IDF transformation to a list of text documents,
    converting them into numerical feature vectors suitable for similarity
    comparison.

    :param corpus: list of str
        A list of text documents (e.g., movie overviews) to be vectorized.

    :return: tuple (TfidfVectorizer, sparse matrix)
        - TfidfVectorizer: The trained TF-IDF vectorizer instance.
        - sparse matrix:
            The resulting TF-IDF matrix of shape (n_samples, n_features).
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(text)
    return vectorizer, tfidf_matrix


def get_recommendations(query, vectorizer, tfidf_matrix, df, top_n=5):
    """
    Get the top N recommendations based on text similarity.

    Computes the cosine similarity between the user query and the TF-IDF
    matrix of the dataset, returning the most similar items.

    :param query: str
        The user's input text query.
    :param vectorizer: TfidfVectorizer
        The trained TF-IDF vectorizer.
    :param tfidf_matrix: scipy.sparse matrix
        The TF-IDF matrix containing vectorized descriptions of all items.
    :param df: pd.DataFrame
        The dataset containing movie titles and descriptions.
    :param top_n: int, optional (default=5)
        The number of top recommendations to return.

    :return: pd.DataFrame
        A DataFrame containing the top recommended movies along with their
        similarity scores.
    """
    query_vec = vectorizer.transform([preprocess_text(query)])
    cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # Get indices of top_n highest similarity scores
    top_indices = cosine_sim.argsort()[-top_n:][::-1]

    # Retrieve top recommendations and reset index for proper ranking
    results = df.iloc[top_indices].copy()
    results['similarity'] = cosine_sim[top_indices]
    results = results.reset_index(drop=True)  # Reset index to start from 0

    return results[['Series_Title', 'similarity']]


def main():
    if len(sys.argv) < 2:
        print("Usage: python recommend.py \"<user query>\"")
        sys.exit(1)

    query = sys.argv[1]

    # Load and preprocess dataset
    df = load_dataset()

    # Build TF-IDF matrix
    vectorizer, tfidf_matrix = build_tfidf_matrix(df['Overview'])

    # Get recommendations
    recommendations = get_recommendations(query, vectorizer, tfidf_matrix, df)

    # Display results
    print("\nTop Recommendations:")
    for i, row in recommendations.iterrows():
        print(f"{i+1}. {row['Series_Title']} "
              f"(Similarity: {row['similarity']:.2f})")


if __name__ == "__main__":
    main()
