from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import traceback

app = Flask(__name__)
CORS(app)

# Load the trained model, vectorizer, and label encoder
model = joblib.load('final_model.pkl')
vectorizer = joblib.load('final_vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form['text']
        print(f"Received text: {data}")  # Debugging print statement
        text_vectorized = vectorizer.transform([data])
        prediction = model.predict(text_vectorized)
        result = label_encoder.inverse_transform(prediction)[0]
        print(f"Prediction result: {result}")  # Debugging print statement
        return jsonify({'prediction': result})
    except KeyError as e:
        error_msg = f"KeyError occurred: {e}"
        print(error_msg)
        print(traceback.format_exc())  # Print full traceback for debugging
        return jsonify({'error': error_msg}), 400
    except ValueError as e:
        error_msg = f"ValueError occurred: {e}"
        print(error_msg)
        print(traceback.format_exc())  # Print full traceback for debugging
        return jsonify({'error': error_msg}), 400
    except Exception as e:
        error_msg = f"An unexpected error occurred: {e}"
        print(error_msg)
        print(traceback.format_exc())  # Print full traceback for debugging
        return jsonify({'error': error_msg}), 500

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(Exception)
def unhandled_exception(e):
    error_msg = f"Unhandled exception: {e}"
    print(error_msg)
    print(traceback.format_exc())  # Print full traceback for debugging
    return jsonify({'error': error_msg}), 500

if __name__ == '__main__':
    app.run(debug=True)
