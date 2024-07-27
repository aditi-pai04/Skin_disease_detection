import pickle
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import numpy as np
import cv2

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the pre-trained CNN model (assuming it's saved as 'model_cnn.h5')
model = load_model('model_cnn.h5')

# Define class labels
class_names = ["Class 1", "Class 2", "Class 3", "Class 4", "Class 5", "Class 6", "Class 7", "Class 8"]

app = Flask(__name__)

# Define the folder to store uploaded images
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def index():
    return render_template('index.html')


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Get the uploaded image
        image_file = request.files["image"]
        if image_file:
            # Preprocess the image
            img = image.load_img(image_file, target_size=(224, 224))  # Adjust size based on your model
            x = image.img_to_array(img)
            x = x / 255.0  # Normalize pixel values
            x = np.expand_dims(x, axis=0)  # Add a batch dimension

            # Make prediction
            prediction = model.predict(x)[0]
            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction) * 100  # Convert to percentage

            return render_template("result.html", prediction=predicted_class, confidence=confidence)
        else:
            return "Please upload an image!"


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
            # Save the uploaded image
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            # You can add logic here to process the uploaded image further
            # (e.g., call an external function or process it within the route)
            prediction = "Image uploaded successfully!"  # Replace with actual processing
            return render_template('result.html', prediction=prediction, image_file=filename)


if __name__ == '__main__':
    app.run(debug=True)
