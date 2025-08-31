import os
import numpy as np
import pandas as pd
import librosa

def get_features(file_path, n_mfcc=40, seconds=4):
    x, sr = librosa.load(file_path, sr=None) 
    target = sr * seconds + 1
    x = np.pad(x, (0, max(0, target - len(x))))[:target]
    mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0).tolist()

def make_dataframe(folder_path, class_by_file_name):
    rows = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".wav"):
            file_path = os.path.join(folder_path, filename)
            rows.append([filename] + get_features(file_path) + [class_by_file_name[filename]])
    cols = ["file_name"] + [f"mfcc_{i+1}" for i in range(40)] + ["class"]
    return pd.DataFrame(rows, columns=cols)