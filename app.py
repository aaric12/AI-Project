# from flask import Flask, render_template, request, jsonify
# import numpy as np
# import pandas as pd
# from tensorflow.keras.models import model_from_json
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# import pickle
# import os

# app = Flask(__name__)

# # Load the trained models and necessary files
# def load_model(model_name):
#     try:
#         # Load model structure (JSON) and weights
#         with open(f'model/{model_name}.json', "r") as json_file:
#             model_json = json_file.read()
#         model = model_from_json(model_json)
#         model.load_weights(f'model/{model_name}_weights.h5')
#         return model
#     except Exception as e:
#         print(f"Error loading model: {e}")
#         return None

# # Load pre-trained models once
# rnn_model = load_model('rnnmodel')
# lstm_model = load_model('lstmmodel')
# ff_model = load_model('ffmodel')

# # Load Label Encoder and Scaler
# le = LabelEncoder()
# scaler = StandardScaler()

# # Preprocess dataset to fit LabelEncoder and Scaler
# crop_dataset = pd.read_csv('dataset/Agriculture In India.csv')
# crop_dataset.fillna(0, inplace=True)
# crop_dataset['State_Name'] = le.fit_transform(crop_dataset['State_Name'])
# crop_dataset['District_Name'] = le.fit_transform(crop_dataset['District_Name'])
# crop_dataset['Season'] = le.fit_transform(crop_dataset['Season'])
# crop_dataset['Crop'] = le.fit_transform(crop_dataset['Crop'])
# X = crop_dataset.drop('Production', axis=1)
# scaler.fit(X)

# # Home route
# @app.route('/')
# def home():
#     return render_template('index.html')

# # Prediction route
# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Get input data from the form
#         state = int(request.form['state'])
#         district = int(request.form['district'])
#         season = int(request.form['season'])
#         crop = int(request.form['crop'])

#         # Prepare input for prediction
#         input_data = np.array([state, district, season, crop]).reshape(1, -1)
#         input_data = scaler.transform(input_data)

#         # Choose model and make prediction
#         model_choice = request.form['model']
#         if model_choice == 'RNN':
#             model = rnn_model
#         elif model_choice == 'LSTM':
#             model = lstm_model
#         else:
#             model = ff_model

#         prediction = model.predict(input_data)
#         predicted_class = np.argmax(prediction)

#         # Interpret the prediction
#         yield_prediction = 'HIGH' if predicted_class == 1 else 'LOW'
#         return jsonify({'prediction': yield_prediction})

#     except Exception as e:
#         return jsonify({'error': str(e)})

# if __name__ == "__main__":
#     app.run(debug=True)
from flask import Flask, request, jsonify
from rapidfuzz import process
from spellchecker import SpellChecker

app = Flask(__name__)

# Sample predefined lists for validation
states = ['Maharashtra', 'Uttar Pradesh', 'Tamil Nadu', 'Karnataka', 'Andhra Pradesh']
districts = ['Pune', 'Agra', 'Chennai', 'Mumbai', 'Bangalore']
seasons = ['Rabi', 'Kharif', 'Zaid']
crops = ['Rice', 'Wheat', 'Sugarcane', 'Cotton', 'Maize']

# Spell checker and fuzzy matching
spell = SpellChecker()

def correct_spelling(input_text, word_list):
    corrected_word = spell.correction(input_text)
    match = process.extractOne(corrected_word, word_list)
    return match[0]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    state = correct_spelling(data['state'], states)
    district = correct_spelling(data['district'], districts)
    season = correct_spelling(data['season'], seasons)
    crop = correct_spelling(data['crop'], crops)
    
    # Here you would add your prediction logic
    prediction = f"Predicted crop yield based on {state}, {district}, {season}, {crop}"
    
    return jsonify({"prediction": prediction})

if __name__ == '__main__':
    app.run(debug=True)
