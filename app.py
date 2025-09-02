from flask import Flask, render_template, request, redirect, flash, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

import numpy as np
import pandas as pd
import librosa
import librosa.display
from src.extract.utils import get_features
    

def make_plots(audio_path: str, out_stem: str):
    y, sr = librosa.load(audio_path, sr=None)
    # Waveform
    wave_path = os.path.join(PLOTS_DIR, f"{out_stem}_wave.png")
    plt.figure(figsize=(8, 2.5))
    librosa.display.waveshow(y, sr=sr)
    plt.title("Waveform")
    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(wave_path, bbox_inches="tight")
    plt.close()
    # Mel-Spectrogram
    mel_path = os.path.join(PLOTS_DIR, f"{out_stem}_mel.png")
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(8, 3))
    librosa.display.specshow(S_dB, sr=sr, x_axis="time", y_axis="mel", fmax=8000)
    plt.title("Mel-Spectrogram (dB)")
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()
    plt.savefig(mel_path, bbox_inches="tight")
    plt.close()
    # relative paths for templates
    wave_rel = f"plots/{out_stem}_wave.png"
    mel_rel  = f"plots/{out_stem}_mel.png"
    return wave_rel, mel_rel


def get_confidence_and_prediction(file_path):

    features = np.array(get_features(file_path)).reshape(1, -1)
    Xp = scaler.transform(pca.transform(features))

    if hasattr(model, "predict_proba"):
        probs_arr = model.predict_proba(Xp)[0]
        pred_idx = int(probs_arr.argmax())
        class_codes = model.classes_
        pred_code = class_codes[pred_idx]
        pred_label = le.inverse_transform([pred_code])[0]
        conf = float(probs_arr[pred_idx])  # in [0,1]
        return pred_label, conf
    else:
        pred_code = model.predict(Xp)[0]
        pred_label = le.inverse_transform([pred_code])[0]
        return pred_label, None  # no confidence


model = joblib.load("models/model.pkl")
pca   = joblib.load("models/pca.pkl")
scaler= joblib.load("models/scaler.pkl")
le    = joblib.load("models/label_encoder.pkl")


UPLOAD_DIR = "uploads"
PLOTS_DIR = os.path.join("static", "plots")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


app = Flask(__name__)
app.secret_key = "secret"


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("audio")
        if not file or not file.filename.lower().endswith(".wav"):
            flash("Please upload a .wav file")
            return redirect(url_for("index"))

        filename = secure_filename(file.filename)
        save_path = os.path.join(UPLOAD_DIR, filename)
        file.save(save_path)

        stem = os.path.splitext(filename)[0]
        wave_rel, mel_rel = make_plots(save_path, stem)
        predicted_label, confidence = get_confidence_and_prediction(save_path)
        predicted_label = "Our prediction for this audio is: " + predicted_label

        return render_template(
            "result.html",
            filename=filename,
            file_url=url_for("serve_upload", filename=filename),
            prediction=predicted_label,
            confidence=confidence,
            wave_url=url_for("static", filename=wave_rel),
            mel_url=url_for("static", filename=mel_rel),
        )
    return render_template("index.html")


@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True)