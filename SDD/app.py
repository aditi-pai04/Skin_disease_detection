import pickle
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import cv2
import pickle
from keras.models import load_model
from keras.preprocessing import image
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import joblib

# import pickle

# # Assuming 'model' is your trained model
# model = 'model_cnn.h5'  # Your trained model object

# # Save the model to a .pkl file
# with open('model_cnn.pkl', 'wb') as file:
#     pickle.dump(model, file)

app = Flask(__name__)
# Define the folder to store uploaded images
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
top3_indices=[]
top3_probabilities=[]
top3_classes=[]

data = pd.read_csv("C:/Users/aditi/OneDrive/Desktop/aiml_el/AIML_EL/SDD/symptoms.csv")
treatment_data = pd.read_csv("C:/Users/aditi/OneDrive/Desktop/aiml_el/AIML_EL/SDD/treatment.csv")
print(treatment_data)
# Separate features (X) and target variable (y)
X = data.drop("Class Label", axis=1)
y = data["Class Label"]

# Initialize the RandomForestClassifier
rf_classifier = joblib.load('C:/Users/aditi/OneDrive/Desktop/aiml_el/AIML_EL/SDD/symptoms_model.joblib')

def get_treatment_for_disease(disease_name):
    # Search for the disease name in the DataFrame
    treatment = treatment_data.loc[treatment_data['Disease Name'] == disease_name, "Don'ts"].values
    print(treatment)
    if len(treatment) > 0:
        return treatment[0]
    else:
        return "Treatment information not found for this disease."
    
def load_custom_model(file_path):
    with open(file_path, 'rb') as file:
        model = pickle.load(open('model_cnn.pkl', 'rb'))
    return model
model = load_model('C:/Users/aditi/OneDrive/Desktop/aiml_el/AIML_EL/SDD/final_model.h5')
print(model)
class_names = {
    0: 'cellulitis',
    1: 'impetigo',
    2: 'athlete-foot',
    3: 'nail-fungus',
    4: 'ringworm',
    5: 'cutaneous-larva-migrans',
    6: 'chickenpox',
    7: 'shingles',
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predictfunc(filepath):
    if request.method == "POST":
        # Check if a filepath is provided
        if filepath:
            img = image.load_img(filepath, target_size=(150, 150))  # Adjust size based on your model
            x = image.img_to_array(img)
            x = x / 255.0
            x = np.expand_dims(x, axis=0)
            prediction = model.predict(x)
            
            print(prediction)
            
            # Get the indices of the top 3 predictions
            top3_indices = np.argsort(prediction[0])[::-1][:3]
            
            # Get the corresponding class names and probabilities
            top3_classes = [class_names[i] for i in top3_indices]
            top3_probabilities = [prediction[0][i] for i in top3_indices]
            return top3_classes, top3_probabilities
        else:
            return "An error occurred. Please try again."
        
@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')
        file = request.files['file']
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return render_template('index.html', message='No selected file')
        if file:
            # Create the upload folder if it doesn't exist
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])
            # Save the uploaded image as "image.png"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'image.png')
            file.save(filepath)
            # Process the image
            predicted_class, confidence = predictfunc(filepath)
            return render_template('result.html', predicted_class=predicted_class, confidence=confidence)

def predict_disease(symptoms,filepath):
    predicted_class, top3_probabilities = predictfunc(filepath)
    print(top3_probabilities)
    max_prob=float('-inf')
    combined_probability=[]
    all=[]
    max_class=' '
    prediction = rf_classifier.predict(symptoms)
    print(prediction)
    for j in range(len(predicted_class)):
        if predicted_class[j]==prediction[0].strip():
            break
    print(j)
    print(top3_probabilities)
    for i in range(len(top3_probabilities)):
        if i==j:
            top3_probabilities[i] = 0.3 * top3_probabilities[i] + 0.7 *0.85
    print(top3_probabilities)
        # print(combined_probability)
        # all.append(combined_probability)
        # print(all)
        # if combined_probability>max_prob:
        #     max_prob=combined_probability
    max_class = predicted_class[np.argmax(top3_probabilities)]
    return max_class

@app.route('/process_symptoms', methods=['POST'])
def symptoms_predict():
    if request.method == 'POST':
        # Get symptom inputs from the HTML form
        print(request.form)
        symptoms = []
        for column in X.columns:
            symptom = request.form[column]
            symptom_value = 1 if symptom.lower() == 'yes' else 0
            symptoms.append(symptom_value)
        symptoms = np.array(symptoms).reshape(1, -1)
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])
            # Save the uploaded image as "image.png"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'image.png')
        # Predict disease
        predicted_disease = predict_disease(symptoms,filepath)
        treatment = get_treatment_for_disease(predicted_disease)
        print(treatment)
        return render_template('final.html', predicted_disease=predicted_disease,  treatment=treatment)

if __name__ == '__main__':
    app.run(debug=True)