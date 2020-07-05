# Importing essential libraries
from flask import Flask, render_template, request
import numpy as np
from tensorflow import keras


# Loading the Sequential model
model = keras.models.load_model("imdb_model.h5")
word_index = np.load("word_index.npy", allow_pickle = True).item()


# Function to encode a text based review into a list of integers
def review_encode(s):

    encoded = [1]    # 1 implies "<START>"

    for word in s:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()] if (word_index[word.lower()] < 88001) else 2)    # vocabulary size is 88000
        else:
            encoded.append(2)    # 2 implies "<UNK>"

    return encoded


app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    
    if request.method == 'POST':

        review = request.form["review"]

        review = review.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace("\"","").strip().split(" ")
        encode = review_encode(review)

        # Make the review 500 words long
        encode = keras.preprocessing.sequence.pad_sequences([encode], value = word_index["<PAD>"], padding = "post", maxlen = 500)

        predict = float(model.predict(encode)[0])

        Sentiment = "Positive" if (predict > 0.5) else "Negative"
        Rating = f"{((predict*1000)//1)/100} / 10"
              
        return render_template('result.html', rating =  Rating, sentiment = Sentiment)


if __name__ == '__main__':
    app.run(debug = True)