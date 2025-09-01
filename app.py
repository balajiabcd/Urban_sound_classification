from flask import Flask, render_template, request, redirect, flash
import os
from werkzeug.utils import secure_filename
from main import predict_sound, check_file

app = Flask(__name__)
app.secret_key = "secret"
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("audio")
        if not file or not file.filename.endswith(".wav"):
            flash("Please upload a .wav file")
            return redirect("/")
        filename = secure_filename(file.filename)
        save_path = os.path.join(UPLOAD_DIR, filename)
        file.save(save_path)
        
        prediction = predict_sound(save_path)
        actural_label = check_file(file.filename)
        flash(f"Predicted sound: {prediction}")
        flash(actural_label)

        return redirect("/")
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
