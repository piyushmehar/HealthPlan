from flask import Flask , render_template,request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl' ,'rb'))

@app.route('/')
def hello():
    return render_template("index1.html")

@app.route('/predict', methods=['POST'])
def predict():
    
    #for putting the details on the html GUI
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)

    return render_template('index1.html', prediction_text='Person should have minimum insurance of : USD {}'.format(output))

if __name__=="__main__":
    app.run()
