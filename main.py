import numpy as np
import pandas as pd
import joblib
from src.extract.utils import get_features


pca = joblib.load("models/pca.pkl")
scaler = joblib.load("models/scaler.pkl")
model = joblib.load("models/model.pkl")
le = joblib.load("models/label_encoder.pkl")


path = 'archive/fold1/7061-6-0-0.wav'
x = get_features(path)
x_arr = np.array(x).reshape(1, -1)


x_trasnsformed = scaler.transform(pca.transform(x_arr))
y_pred = model.predict(x_trasnsformed)
y_pred_label = le.inverse_transform(y_pred)
print(y_pred_label)