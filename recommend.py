import pandas as pd


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
    :return: str
    """
    return query.lower()  # Convert to lowercase


def main():
    df = load_dataset()
    print(df.head())


if __name__ == "__main__":
    main()
