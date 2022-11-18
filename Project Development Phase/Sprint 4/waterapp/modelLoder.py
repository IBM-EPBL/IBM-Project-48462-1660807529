import pickle
import numpy as np


def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 6)
    loaded_model = pickle.load(open("wqi.pkl", "rb"))
    result = loaded_model.predict(to_predict)
    return result
