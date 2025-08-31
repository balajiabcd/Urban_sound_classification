import os
import numpy as np
import pandas as pd
import librosa
from src.extract.utils import get_features, make_dataframe



os.makedirs("archive_dataframes", exist_ok=True)
meta_data = pd.read_csv("archive/UrbanSound8K.csv")
class_by_file_name = meta_data.set_index("slice_file_name")["class"].to_dict()



archive = "archive"
for folder in os.listdir(archive):
    if "fold" in folder:
        folder_path = os.path.join(archive, folder)
        df = make_dataframe(folder_path, class_by_file_name)
        df.to_csv(f"archive_dataframes/{folder}_df_file.csv", index=False)
        print(df.shape, f"{folder} saving was done")