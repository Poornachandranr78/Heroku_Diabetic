# Load the Random Forest CLassifier model
import pickle
filename = 'diabetes-prediction-rfc-models.pkl'
classifier = pickle.load(open(filename, 'rb'))

from flask import Flask,render_template,request
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        try:
            preg=int(request.form['pregnancies'])
            gul=int(request.form['glucose'])
            bp=int(request.form['bloodpressure'])
            st=int(request.form['skinthickness'])
            insulin=int(request.form['insulin'])
            bmi=float(request.form['bmi'])
            dpf=float(request.form['dpf'])
            age=int(request.form['age'])
            df=np.array([[preg,gul,bp,st,insulin,bmi,dpf,age]])
            mypredict=classifier.predict(df)
            return render_template('result.html',Prediction=mypredict)
        except Exception as e:
            print('The Exception message is: ', e)
            return 'something is wrong from your side'
if __name__ == '__main__':
	app.run(port=8500,debug=True)