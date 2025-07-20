from flask import Flask, render_template, request, send_from_directory
import os
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
MODEL_PATH = 'model/waste_classifier.h5'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
CLASS_NAMES = ['Biodegradable', 'Non-Biodegradable']

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    print("❌ Model loading failed:", e)
    model = None

# Ensure uploads folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Image preprocessing
def preprocess_image(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print("❌ Image error:", e)
        return None

# Route to serve uploaded file
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Main route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            img_array = preprocess_image(filepath)
            if model is None or img_array is None:
                return render_template('index.html', result="Model or image error", error=True)

            prediction = model.predict(img_array)[0]
            predicted_class = CLASS_NAMES[np.argmax(prediction)]
            confidence = round(float(np.max(prediction)) * 100, 2)

            return render_template('index.html',
                                   result=predicted_class,
                                   confidence=confidence,
                                   image_file=filename,
                                   class_names=CLASS_NAMES,
                                   class_probs=[round(float(p) * 100, 2) for p in prediction])
        else:
            return render_template('index.html', result="Invalid file", error=True)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
