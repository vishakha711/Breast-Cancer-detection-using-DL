from flask import Flask, request, render_template_string
import numpy as np
import pickle
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model and scaler
model = load_model('model.h5')
scaler = pickle.load(open('scalers.pkl', 'rb'))

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Breast Cancer Detection</title>
  <style>
    body {
      font-family: 'Poppins', sans-serif;
      background: linear-gradient(135deg, #e0f7fa, #bbdefb);
      text-align: center;
      color: #333;
      padding: 20px;
    }
    .container {
      background: white;
      padding: 25px;
      margin: auto;
      width: 55%;
      box-shadow: 0 4px 15px rgba(0,0,0,0.2);
      border-radius: 15px;
    }
    h1 {
      color: #007bff;
      margin-bottom: 20px;
    }
    form input {
      width: 80%;
      padding: 8px;
      margin: 6px 0;
      border: 1px solid #ccc;
      border-radius: 6px;
      font-size: 14px;
    }
    .btn {
      background: #007bff;
      color: white;
      padding: 10px 20px;
      border: none;
      border-radius: 8px;
      cursor: pointer;
      font-size: 16px;
      margin-top: 10px;
    }
    .btn:hover {
      background: #0056b3;
    }
    .result {
      margin-top: 20px;
      font-weight: bold;
      font-size: 20px;
      color: #004d40;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>🩺 Breast Cancer Detection</h1>
    <p>Enter all 30 feature values below:</p>
    <form action="/" method="post">
      {% for i in range(30) %}
        <input type="text" name="feature{{ i }}" placeholder="Feature {{ i+1 }}" required><br>
      {% endfor %}
      <button type="submit" class="btn">Predict</button>
    </form>

    {% if prediction_text %}
      <div class="result">{{ prediction_text }}</div>
    {% endif %}
  </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_text = None
    if request.method == 'POST':
        try:
            # Get all 30 features from form
            features = [float(request.form[f'feature{i}']) for i in range(30)]
            input_data = np.array(features).reshape(1, -1)

            # Standardize the input
            std_data = scaler.transform(input_data)

            # Predict using model
            prediction = model.predict(std_data)
            label = 1 if prediction > 0.5 else 0

            # Show result
            if label == 1:
                prediction_text = "✅ The Breast Cancer is Benign (Non-Cancerous)"
            else:
                prediction_text = "⚠️ The Breast Cancer is Malignant (Cancerous)"
        except Exception as e:
            prediction_text = f"Error: {str(e)}"

    return render_template_string(HTML_PAGE, prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
