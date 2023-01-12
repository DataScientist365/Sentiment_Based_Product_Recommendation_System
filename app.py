from email import header
from operator import index
from flask import Flask, request, render_template, jsonify
from model import *


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def prediction():

    user = request.form['userName']

    user = user.lower()
    items = top_Recommendations(user)

    if(not(items is None)):
        print(f"retrieving items....{len(items)}")
        print(items)

        return render_template("index.html", column_names=items.columns.values, row_data=list(items.values.tolist()), zip=zip)
    else:
        return render_template("index.html", message="User Name doesn't exists, No product recommendations at this point of time!")

if __name__ == '__main__':
    app.run()