import streamlit as st
import pandas as pd
import joblib
from preprocess import preprocess_data

# Load the model
def load_model(model_path):
    """
    Load a trained model from a file.

    Args:
        model_path (str): Path to the model file.

    Returns:
        model: Loaded model object.
    """
    return joblib.load(model_path)

# Predict lead scores
def predict_scores(model, data):
    """
    Predict lead scores using the trained model.

    Args:
        model: Trained model object.
        data (DataFrame): Preprocessed input data.

    Returns:
        Series: Predicted probabilities for conversion.
    """
    return model.predict_proba(data)[:, 1]

# Streamlit app
st.title("Lead Scoring System")
st.write("Rank potential leads based on their likelihood of conversion.")

# File upload
data_file = st.file_uploader("Upload a CSV file with lead data", type=["csv"])

if data_file is not None:
    # Load data
    input_data = pd.read_csv(data_file)
    st.write("Uploaded Data:", input_data.head())

    # Preprocess data
    feature_columns = ['Age', 'Clicks', 'WebsiteVisits', 'LocationScore']
    try:
        preprocessed_data = preprocess_data(data_file, feature_columns)
        st.write("Preprocessed Data:", preprocessed_data.head())

        # Load the model
        model_path = "models/lead_model.pkl"
        model = load_model(model_path)

        # Predict scores
        scores = predict_scores(model, preprocessed_data[feature_columns])
        input_data['Conversion Probability'] = scores

        # Display results
        st.write("Lead Scoring Results:")
        st.dataframe(input_data)

        # Download option
        csv = input_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Scoring Results as CSV",
            data=csv,
            file_name="lead_scoring_results.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error in processing the data: {e}")
