from flask import Flask, request, jsonify
from flask_restful import Api, Resource
import logging
import pickle
from process_request.processing import predict_data, features_prediction_str

## TODO:
#   better status code
#   error handling


# setup prediction logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('predictions.csv')
file_handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(file_handler)

# load Model
model = pickle.load(open('./model/iris.sav', 'rb'))

# setup flask application
app = Flask(__name__)
api = Api(app)


class Predict(Resource):
    def get(self):
        request_data = request.get_json()
        prediction = predict_data(model, request_data)
        feature_values = ', '.join([str(i) for i in request_data.values()])
        logger.info(features_prediction_str(request_data, prediction))
        return jsonify(prediction)


# add enpoint(s)
api.add_resource(Predict, '/predict')

# run application on port 5000
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
