import streamlit as st
import pandas as pd
import joblib
from src.pipeline.prediction_pipeline import PredictionPipeline
import numpy as np

model_path = 'artifacts/model_trainer/model.pkl'
preprocessor_path = 'artifacts/data_transformation/preprocessor.pkl'

try:
    loaded_model = joblib.load(model_path)
    loaded_preprocessor = joblib.load(preprocessor_path)
    print(f"Model loaded successfully from {model_path}")
    print(f'Preprocessor loaded successfully from {preprocessor_path}')
except FileNotFoundError:
    print(f"Error: Model file '{model_path}' not found.")
    loaded_model = None
    loaded_preprocessor = None
except Exception as e:
    print(f"Error loading model: {e}")
    loaded_model = None
    loaded_preprocessor = None

def predictions(comment):
    if loaded_model is None:
        return "Model not loaded"
    
    custom_data = {
        'comment': [comment],
        
    }

    input_df = pd.DataFrame(custom_data)

    prediction_pipeline = PredictionPipeline()

    # Make a prediction
    prediction = prediction_pipeline.predict(input_df)

    return prediction

def main():
    st.title('CyberBullying Detection')
    st.header("Please enter the data")

    text = st.text_area("Enter the text for prediction")

    if st.button('Predict'):
        try:
            result = predictions(text)
            st.write(f"Raw Prediction Result: {result}")

            # Assuming result is a string label
            if result == 'hate':
                st.write("Prediction: Cyberbully")
            elif result == 'normal':
                st.write("Prediction: Not Cyberbully")
            else:
                st.write("Error: Unexpected result format.")
        except Exception as e:
            st.write(f"Error during prediction: {e}")

if __name__ == '__main__':
    main()
