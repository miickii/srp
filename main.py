from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from tensorflow.keras.models import load_model


digit_model = load_model('digit_model.h5')
doodle_model = load_model('doodle_model2.h5')

#labels = ["penguin", "apple", "airplane", "tree", "pan", "wine glass", "dog", "headphones", "carrot", "bridge", "helicopter", "cactus", "scissors", "bed"]
#labels2_names = ["apple", "tree", "pizza", "eiffel_tower", "donut", "fish", "wine_glass", "dog", "smiley", "carrot", "t_shirt", "cactus", "bed"]
doodle_labels = ["🍎", "🌳", "🍕", "🗼", "🍩", "🐟", "🍷", "🐕", "🙂", "🥕", "👕", "🌵", "🛏️"]
digit_labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173", "https://miickii.github.io/srp-frontend"], allow_headers=["Content-Type"], methods=["GET", "POST"])

def most_likely(prediction, labels):
    sorted_indices = np.argsort(prediction)
    top_pred_idx, alt_pred_idx = sorted_indices[0][-1], sorted_indices[0][-2]
    
    return {"top": [labels[top_pred_idx], int(prediction[0][top_pred_idx]*100)], "alt": [labels[alt_pred_idx], int(prediction[0][alt_pred_idx]*100)]} 

@app.route("/predict", methods=['POST'])
def predict():
    result = ""

    is_doodle = request.json['doodle']
    pixels = request.json['pixels']
    pixels = np.array(pixels)

    if is_doodle:
        prediction = doodle_model.predict(pixels.reshape((1, 784)))
        result = most_likely(prediction, doodle_labels)
    else:
        prediction = digit_model.predict(pixels.reshape((1, 28, 28)))
        result = most_likely(prediction, digit_labels)

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
