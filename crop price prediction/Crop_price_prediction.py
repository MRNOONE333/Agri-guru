import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the trained model
model = joblib.load('newrfmod.pkl')

# Define the Flask API endpoint
@app.route('/', methods=['GET'])
def predict():
    com = int(request.args.get('com'))
    state = int(request.args.get('state'))
    district = int(request.args.get('district'))
    market = int(request.args.get('market'))
    month = int(request.args.get('month'))
    
    user_input = [[com, state, district, market, month]]
    result=model.predict(user_input)

    response = {
        'price': result[0],
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run()