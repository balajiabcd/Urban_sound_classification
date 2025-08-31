import argparse
from pathlib import Path
import joblib, pandas as pd
from sklearn.metrics import f1_score, accuracy_score

from .data_io import load_archive_dataframes
from .preprocess import split_xy, make_splits, fit_transform_pca_scaler, label_encode
from .models import build_candidates
from .evaluate import pca_elbow_plot, save_confusion, write_metrics, get_metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive_df_dir", required=True)     # e.g., archive_dataframes
    ap.add_argument("--target", required=True)             # e.g., classID
    ap.add_argument("--models_dir", default="models")
    ap.add_argument("--reports_dir", default="static")
    ap.add_argument("--pca_components", type=int, default=20)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--val_size", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    models_dir = Path(args.models_dir)
    reports_dir = Path(args.reports_dir)
    figs_dir = reports_dir / "figures"
    metrics_csv = reports_dir / "metrics.csv"

    # 1) Load & split
    df = load_archive_dataframes(args.archive_df_dir)
    X, y = split_xy(df, args.target)
    X_tr, X_va, X_te, y_tr_raw, y_va_raw, y_te_raw = make_splits(X, y, args.test_size, args.val_size, args.seed)

    # 2) Encode + PCA + Scale
    y_tr, y_va, y_te, le = label_encode(y_tr_raw, y_va_raw, y_te_raw)
    X_tr_s, X_va_s, X_te_s, pca, scaler = fit_transform_pca_scaler(X_tr, X_va, X_te, n_components=args.pca_components)

    # plots
    pca_elbow_plot(pca, figs_dir / "pca_elbow.png")

    # 3) Train candidates → pick best by val F1
    candidates = build_candidates(seed=args.seed)
    records = []; best = (None, -1.0, None, None)  # (name, f1, model, y_pred_va)

    for name, model in candidates.items():
        model.fit(X_tr_s, y_tr)
        y_va_pred = model.predict(X_va_s)
        metrics = get_metrics(name, y_va, y_va_pred)
        records.append(metrics)
        if metrics["F1_Score"] > best[1]:
            best = (name, metrics["F1_Score"], model)

    write_metrics(records, metrics_csv)

    best_name, best_f1, best_model = best
    print(f"Best on validation: {best_name} | F1_score_val={best_f1:.3f}")

    # 4) Final test evaluation
    y_te_pred = best_model.predict(X_te_s)
    test_F1_score = f1_score(y_te, y_te_pred, average="macro")
    test_accuracy = accuracy_score(y_te, y_te_pred)
    print(f"test_accuracy={test_accuracy:.3f} | test_F1_score={test_F1_score:.3f}")

    save_confusion(y_te, y_te_pred, figs_dir / "confusion_matrix.png", title=f"{best_name} Confusion")

    # 5) Save artifacts (PKL)
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, models_dir / "model.pkl")
    joblib.dump(pca, models_dir / "pca.pkl")
    joblib.dump(scaler, models_dir / "scaler.pkl")
    joblib.dump(le, models_dir / "label_encoder.pkl")

    # append test row to metrics
    pd.concat([
        pd.read_csv(metrics_csv),
        pd.DataFrame([{"model": best_name, "test_accuracy": test_accuracy, "test_F1_score": test_F1_score}])
    ], ignore_index=True).to_csv(metrics_csv, index=False)

if __name__ == "__main__":
    main()
