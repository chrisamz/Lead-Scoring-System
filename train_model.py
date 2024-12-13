import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
import joblib

# Function to load data
def load_data(filepath):
    """Loads the dataset from the given file path."""
    return pd.read_csv(filepath)

# Function to preprocess data
def preprocess_data(data):
    """Prepares features and target variables for training."""
    X = data[['Age', 'Clicks', 'WebsiteVisits', 'LocationScore']]
    y = data['Outcome']
    return X, y

# Function to train the model
def train_model(X_train, y_train):
    """Trains a Gradient Boosting model on the given data."""
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    """Evaluates the trained model on test data."""
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

# Main execution flow
def main():
    # Load dataset
    filepath = 'data/example_data.csv'  # Update this path as needed
    data = load_data(filepath)

    # Preprocess data
    X, y = preprocess_data(data)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

    # Save the model
    model_path = 'models/lead_model.pkl'
    joblib.dump(model, model_path)
    print(f"Model saved at: {model_path}")

if __name__ == "__main__":
    main()
