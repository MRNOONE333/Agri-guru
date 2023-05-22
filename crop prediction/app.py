import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the trained model
model = joblib.load('randommodel.pkl')

# Define a function to make predictions
def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    user_input = [[N,P,K,temperature,humidity,ph,rainfall]]
    crop_names=['apple','banana','blackgram','chickpea','coconut','coffee','cotton','grapes','jute','kidneybeans','lentil','maize','mango','mothbeans','mungbean','muskmelon','orange','papaya','pigeonpeas','pomegranate','rice','watermelon']
    crop_prob_dict = dict(zip(crop_names, model.predict_proba(user_input)[0]))
    top_crops = sorted(crop_prob_dict, key=crop_prob_dict.get,reverse = True)[:3]
    return top_crops

# Define the Flask API endpoint
@app.route('/', methods=['GET'])
def predict():
    N = float(request.args.get('n'))
    P = float(request.args.get('p'))
    K = float(request.args.get('k'))
    temperature = float(request.args.get('temp'))
    humidity = float(request.args.get('humid'))
    ph = float(request.args.get('ph'))
    rainfall = float(request.args.get('rain'))

    result = predict_crop(N, P, K, temperature, humidity, ph, rainfall)
    response = {
        'crop1': result[0],
        'crop2': result[1],
        'crop3': result[2],
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run()
