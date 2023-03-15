from flask import Flask, request, jsonify
from flask_cors import cross_origin
import numpy as np
from tensorflow.keras.models import load_model


digit_model_small = load_model('digit_model_small.h5')
digit_model_large = load_model('digit_model_large.h5')
digit_model = digit_model_small

doodle_model_small = load_model('doodle_model2.h5') # 8000 training examples
doodle_model_medium = load_model('doodle_model_medium.h5') # 18000 training examples
doodle_model_large = load_model('doodle_model_large.h5') # 70000 training examples
doodle_model = doodle_model_small

tegning_labeler = ["Æble", "Træ", "Pizza", "Eiffeltårn", "Donut", "Fisk", "Vinglas", "Hund", "Smiley", "Gulerod", "T-shirt", "Kaktus", "Seng"]
ciffer_labeler = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

app = Flask(__name__)

def most_likely(prediction, labels):
    sorted_indices = np.argsort(prediction)
    top_pred_idx, alt_pred_idx = sorted_indices[0][-1], sorted_indices[0][-2]
    
    return {"top": [labels[top_pred_idx], int(prediction[0][top_pred_idx]*100)], "alt": [labels[alt_pred_idx], int(prediction[0][alt_pred_idx]*100)]} 

@app.route("/predict", methods=['POST'])
@cross_origin()
def predict():
    result = ""

    is_doodle = request.json['doodle']
    pixels = request.json['pixels']
    pixels = np.array(pixels)

    if is_doodle:
        prediction = doodle_model.predict(pixels.reshape((1, 784)))
        result = most_likely(prediction, tegning_labeler)
    else:
        prediction = digit_model.predict(pixels.reshape((1, 28, 28)))
        result = most_likely(prediction, ciffer_labeler)

    return jsonify(result)

@app.route("/change-model", methods=['POST'])
@cross_origin()
def change_model():
    global doodle_model, digit_model
    is_digit = request.json['digitModel']
    new_model = request.json['newModel']

    if new_model == 1:
        if is_digit:
            digit_model = digit_model_small
        else:
            doodle_model = doodle_model_small
    elif new_model == 2:
        doodle_model = doodle_model_medium
    elif new_model == 3:
        if is_digit:
            digit_model = digit_model_large
        else:
            doodle_model = doodle_model_large

    return jsonify("ok")


if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
