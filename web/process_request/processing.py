import numpy as np
import logging

logger = logging.getLogger(__name__)
file_handler = logging.FileHandler('./errors.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s: %(message)s'))
logger.addHandler(file_handler)

def _validate_request(request_data):
    """checks for correct feature labels and value types"""
    valid_features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    request_values = request_data.values()

    try:
        for feature in valid_features:
            request_data[feature]
    except KeyError as e:
        logger.error('feature not found in request data: {}'.format(e))

    try:
        for value in request_values:
            float(value)
    except ValueError as e:
        logger.error('could not conver to float: {}'.format(e))


def _preprocess(request_data):
    """gets the features and data from the request data and shapes it correctly into a numpy array"""
    features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    feature_data = np.array([request_data.get(feature) for feature in features])
    feature_data = feature_data.reshape(1, len(features))
    return feature_data


def _pretty_json(prediction):
    """takes prediction and formats it into a user-friendly dictionary"""
    prediction_labels = ['setosa', 'versicolor', 'virginica']
    prediction = [prob[0][1] for prob in prediction]
    prediction_dict = {label: proba for label, proba in zip(prediction_labels, prediction)}
    return prediction_dict

def features_prediction_str(request_data, prediction):
    features = ', '.join([str(i) for i in request_data.values()])
    prediction = ', '.join([str(i) for i in prediction.values()])
    features_prediction = features + ', ' + prediction
    return features_prediction

def predict_data(model, request_data):
    """
    Takes the request data:
        - validates
        - predicts
        - formats into pretty dictionary
    """
    _validate_request(request_data)
    feature_data = _preprocess(request_data)
    prediction = model.predict_proba(feature_data)
    prediction = _pretty_json(prediction)
    return prediction
