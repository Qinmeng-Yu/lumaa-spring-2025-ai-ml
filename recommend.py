import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def load_dataset(filepath="dataset/imdb_top_1000.csv"):
    """
    Load the dataset from the given file path
    :param filepath: str
    :return: pd.DataFrame
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


def main():
    df = load_dataset()
    vectorizer, tfidf_matrix = build_tfidf_matrix(df['Overview'])


if __name__ == "__main__":
    main()
