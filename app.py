from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load('fertilizer.pkl')

@app.route('/', methods=['GET'])
def predict():
     temp = int(request.args.get('temp'))
     humid = int(request.args.get('humid'))
     moist = int(request.args.get('moist'))
     soil = int(request.args.get('soil'))
     crop = int(request.args.get('crop'))
     n = int(request.args.get('n'))
     p = int(request.args.get('p'))
     k = int(request.args.get('k'))
     user_input = [[temp,humid,moist,soil,crop,n,p,k]]
     pred = model.predict(user_input)
     print(pred)
     response = {
         'prediction':pred[0]
     }
     return jsonify(response)


if __name__ == '__main__':
     app.run()