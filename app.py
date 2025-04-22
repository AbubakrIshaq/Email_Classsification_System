from flask import Flask, request, jsonify, render_template
from models import SpamClassifier
import os

app = Flask(__name__)
classifier = SpamClassifier()

# Load pre-trained model
if not classifier.load_model():
    print("Warning: Pre-trained model not found. Please train the model first.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_email():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    email_text = data['text']
    result = classifier.predict(email_text)
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)