import os
import numpy as np
import pandas as pd
import joblib
from src.extract.utils import get_features


path = 'sample_data/fold2/7383-3-0-1.wav'

def predict_sound(path):
    x = get_features(path)
    x_arr = np.array(x).reshape(1, -1)

    pca = joblib.load("models/pca.pkl")
    scaler = joblib.load("models/scaler.pkl")
    model = joblib.load("models/model.pkl")
    le = joblib.load("models/label_encoder.pkl")

    feature_names = [f"mfcc_{i+1}" for i in range(x_arr.shape[1])]
    x_df = pd.DataFrame(x_arr, columns=feature_names)

    x_trasnsformed = scaler.transform(pca.transform(x_df))
    y_pred = model.predict(x_trasnsformed)
    y_pred_label = le.inverse_transform(y_pred)
    return y_pred_label[0]

if __name__ == "__main__":
    path = 'sample_data/fold2/7383-3-0-1.wav'
    y_pred_label = predict_sound(path)
    filename = os.path.basename(path)

    data = pd.read_csv("sample_data/UrbanSound8K.csv")
    class_by_file_name = data.set_index("slice_file_name")["class"].to_dict()

    if filename in class_by_file_name:
        actural_label = class_by_file_name[filename]
        print( "we have trained on this audio file. The actual sound is: ",actural_label)
    else:
        actural_label = "This is a new audio file. We don't have actual label"
        print(actural_label) 

    print("The prediction for this audio file is: ", y_pred_label)

