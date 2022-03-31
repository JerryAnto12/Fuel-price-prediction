import numpy as np
import pandas as pd
from flask import Flask, render_template,request
import pickle#Initialize the flask App
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():

    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    return render_template('index.html', prediction_text='fuel price for kilometer driven is :{}'.format(prediction))
if __name__ == "__main__":
    app.run(debug=True)