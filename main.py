from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from tensorflow.keras.models import load_model


digit_model = load_model('digit_model.h5')
doodle_model = load_model('doodle_model2.h5')

labels = ["penguin", "apple", "airplane", "tree", "pan", "wine glass", "dog", "headphones", "carrot", "bridge", "helicopter", "cactus", "scissors", "bed"]
labels2_names = ["apple", "tree", "pizza", "eiffel_tower", "donut", "fish", "wine_glass", "dog", "smiley", "carrot", "t_shirt", "cactus", "bed"]
labels2 = ["🍎", "🌳", "🍕", "🗼", "🍩", "🐟", "🍷", "🐕", "🙂", "🥕", "👕", "🌵", "🛏️"]


app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"], allow_headers=["Content-Type"], methods=["GET", "POST"])

@app.route("/predict", methods=['POST'])
def predict():
    result = ""

    is_doodle = request.json['doodle']
    pixels = request.json['pixels']
    pixels = np.array(pixels)

    if is_doodle:
        prediction = doodle_model.predict(pixels.reshape((1, 784)))
        result = labels2[np.argmax(prediction)]
    else:
        prediction = digit_model.predict(pixels.reshape((1, 28, 28)))
        result = str(np.argmax(prediction))

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
