# Plant Detection ML Model
This project implements a machine learning model for detecting plant species from images. It uses a pre-trained TensorFlow Keras model and is hosted on a Flask server with an API for predictions.

# Features
Predict plant species from an image URL.
REST API for easy integration.
# Prerequisites
Python 3.7+
Required libraries:
```bash

pip install tensorflow flask pillow numpy requests flask-cors
```
Project Structure
app.py: Main Flask application.
KS.h5: Pre-trained TensorFlow Keras model.
API Endpoint
POST /predict/
Request: Send a POST request with a form field url containing the image URL.
Example Request:
```bash

curl -X POST -F "url=https://example.com/image.jpg" http://localhost:5000/predict/
```

Response:
200 OK: Returns the predicted plant species.
json
```
{
  "class": "Aloevera"
}
```
400 Bad Request: If no URL is provided.

# Setup and Usage

Clone the repository:
```bash

git clone https://github.com/bansalmohit123/model.git
```

Place the model file (KS.h5) in the project directory.

Run the application:

```bash

python m2.py
```
The Flask server runs on http://localhost:5000.
