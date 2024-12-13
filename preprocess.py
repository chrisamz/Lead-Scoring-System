import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_data(filepath):
    """
    Loads the dataset from a CSV file.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        DataFrame: Loaded dataset as a Pandas DataFrame.
    """
    return pd.read_csv(filepath)

def clean_data(data):
    """
    Cleans the dataset by handling missing values and duplicates.

    Args:
        data (DataFrame): Raw dataset.

    Returns:
        DataFrame: Cleaned dataset.
    """
    # Drop duplicate rows
    data = data.drop_duplicates()

    # Fill or drop missing values
    data = data.fillna(data.median())

    return data

def scale_features(data, feature_columns):
    """
    Scales the feature columns to a 0-1 range using Min-Max scaling.

    Args:
        data (DataFrame): Dataset with feature columns.
        feature_columns (list): List of columns to scale.

    Returns:
        DataFrame: Dataset with scaled feature columns.
    """
    scaler = MinMaxScaler()
    data[feature_columns] = scaler.fit_transform(data[feature_columns])
    return data

def preprocess_data(filepath, feature_columns):
    """
    Complete preprocessing pipeline: loading, cleaning, and scaling the data.

    Args:
        filepath (str): Path to the CSV file.
        feature_columns (list): List of feature columns to scale.

    Returns:
        DataFrame: Preprocessed dataset.
    """
    data = load_data(filepath)
    data = clean_data(data)
    data = scale_features(data, feature_columns)
    return data

if __name__ == "__main__":
    # Example usage
    filepath = "data/example_data.csv"
    feature_columns = ['Age', 'Clicks', 'WebsiteVisits', 'LocationScore']

    preprocessed_data = preprocess_data(filepath, feature_columns)
    print(preprocessed_data.head())
