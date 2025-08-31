import argparse
import joblib
import pandas as pd
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_dir", default="models")
    ap.add_argument("--features_csv", required=True)  # same columns as train X (numeric)
    ap.add_argument("--out_csv", default="reports/predictions.csv")
    args = ap.parse_args()

    md = Path(args.models_dir)
    model = joblib.load(md / "model.pkl")
    pca   = joblib.load(md / "pca.pkl")
    sc    = joblib.load(md / "scaler.pkl")
    le    = joblib.load(md / "label_encoder.pkl")

    df = pd.read_csv(args.features_csv)
    X = df.select_dtypes(include="number")
    Xp = sc.transform(pca.transform(X))
    y_pred = model.predict(Xp)
    labels = le.inverse_transform(y_pred)

    out = pd.DataFrame({"prediction": labels})
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Saved predictions → {args.out_csv}")

if __name__ == "__main__":
    main()
