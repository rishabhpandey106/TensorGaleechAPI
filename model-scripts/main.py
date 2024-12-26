import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Path to the locally saved TensorFlow model
MODEL_PATH = "converted_model.h5"

# Load the TensorFlow model from the local file system
model = tf.keras.models.load_model(MODEL_PATH)

# Load the SentenceTransformer model for generating embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

@app.route("/", methods=["POST"])
def predict():
    try:
        # Parse the incoming request
        data = request.json
        message = data.get("message", "")
        print(f"Received message: {message}")

        if not message:
            return jsonify({"error": "Message is required"}), 400

        # Generate embeddings using SentenceTransformer
        embedding = embedding_model.encode([message])

        # Convert embedding to a tensor
        input_tensor = tf.convert_to_tensor(embedding, dtype=tf.float32)

        # Predict using the TensorFlow model
        prediction = model.predict(input_tensor)
        prediction_score = float(prediction[0][0])  # Assuming single output node

        return jsonify({
            "prediction": prediction_score,
            "note": "1 is very toxic/profane, 0 is not profane at all",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
