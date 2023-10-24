#this file is app.py

from flask import Flask , render_template,request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('F:\MyDS\projects\insurance price\model.pkl' ,'rb'))

@app.route('/')
def hello():
    return render_template("home.html")

@app.route('/home.html')
def home():
    return render_template("home.html")

@app.route('/blog.html')
def blog():
    return render_template("blog.html")

@app.route('/contact.html')
def contact():
    return render_template("contact.html")

@app.route('/predict.html')
def index():
    return render_template("predict.html")

@app.route('/predict.html', methods=['POST'])
def predict():
    
    #for putting the details on the html GUI
    age	 = request.form.get('age')
    sex = request.form.get('sex')
    bmi = request.form.get('bmi')
    children = request.form.get('children')
    smoker = request.form.get('smoker')
    region	 = request.form.get('region')
    features = [age,sex,bmi,children,smoker,region]
    int_features = [eval(i) for i in features]
    final_features = [np.array(features)]
    final_features =  np.asarray(int_features)
    final_features_reshape =  final_features.reshape(1,-1)
    prediction = model.predict(final_features_reshape)
    output = round(prediction[0], 2)

    return render_template('predict.html', prediction_text='Person should have minimum insurance of : {}'.format(output))



if __name__=="__main__":
    app.run()
