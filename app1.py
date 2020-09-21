import flask
from flask import request
import numpy as np
from tensorflow.keras.models import load_model

# instantiate flask
app = flask.Flask(__name__)

with open("dataset.txt", 'r', encoding='utf-8') as f:
    data_1 = f.read().lower()


@app.route("/")
def hello():
    return "hello"


# define a predict function as an endpoint
@app.route("/predict", methods=["GET","POST"])
def predict():
    # print(data)
    text = " "
    content = request.get_json(silent=True)
    # print(content)
    x = content["text"]
    maxlen = 45
    chars = sorted(list(set(data_1)))
    char_indices = dict((char, chars.index(char)) for char in chars)
    model = load_model('final_three_lstm1.h5')
    # x = "with clytia he no longer was received than while he was a man of wealth believed balls , concerts , op'ras , tournaments , and plays "
    start_idx = np.random.randint(0, len(x) - maxlen - 1)
    new_sonnet = data_1[start_idx : start_idx +maxlen]
    # new_sonnet = x
    # sys.stdout.write(new_sonnet)
    print(new_sonnet)
    for i in range(600):
        # Vectorize generated text
        sampled = np.zeros((1, maxlen, len(chars)))
        for j, char in enumerate(new_sonnet):
            sampled[0, j, char_indices[char]] = 1
        preds = model.predict(sampled, verbose=0)[0]
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / 0.5
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        pred_idx = np.argmax(probas)

        next_char = chars[pred_idx]

        # Append predicted character to seed text
        new_sonnet += next_char
        new_sonnet = new_sonnet[1:]
        x += next_char
        print(next_char,end = " ")


    return {"sonnet" : x}

# # start the flask app, allow remote connections
if __name__ == '__main__':
    app.run(debug=True)

# app.run(host='0.0.0.0')