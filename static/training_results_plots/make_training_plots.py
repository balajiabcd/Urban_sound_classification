import os, re
import pandas as pd, numpy as np, matplotlib.pyplot as plt

# Paths
HERE = os.path.dirname(__file__)
METRICS = os.path.join(HERE, "..", "metrics.csv")
OUT = HERE

df = pd.read_csv(METRICS).dropna(subset=["F1_Score"])
df = df.sort_values("F1_Score", ascending=False)

# 1) Top 5 & Bottom 5
x = pd.concat([df.head(5), df.tail(5)])
plt.bar(range(len(x)), x["F1_Score"])
plt.xticks(range(len(x)), x["model"], rotation=45, ha="right")
plt.ylabel("F1-Score"); plt.title("Top 5 and Bottom 5 Models")
plt.tight_layout()
plt.savefig(os.path.join(OUT,"bar_best5_worst5.png"), dpi=200); plt.close()

# 2) KNN neighbors vs F1
rows=[(m.split("_")[1],int(m.split("_")[2][1:]),f1) for m,f1 in zip(df["model"],df["F1_Score"]) if m.startswith("knn_")]
knn=pd.DataFrame(rows,columns=["metric","k","F1"])
plt.figure()
for met in ["cosine","euclidean","manhattan"]:
    d=knn[knn.metric==met].sort_values("k")
    plt.plot(d["k"], d["F1"], marker="o", label=met)
plt.xlabel("k"); plt.ylabel("F1-Score"); plt.title("KNN Performance by Metric"); plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT,"line_knn_neighbors_f1.png"), dpi=200); plt.close()

# 3) Best 3 vs Worst 2
s=pd.concat([df.head(3), df.tail(2)])
plt.bar(range(len(s)), s["F1_Score"])
plt.xticks(range(len(s)), s["model"], rotation=30, ha="right")
plt.ylabel("F1-Score"); plt.title("Best 3 vs Worst 2")
plt.tight_layout()
plt.savefig(os.path.join(OUT,"bar_summary_best3_worst2.png"), dpi=200); plt.close()

print("Saved plots in", OUT)


# ==== Step 4 — Evaluation plots (append to bottom) ====
df_eval = pd.read_csv(METRICS).dropna(subset=["F1_Score"])
df_eval = df_eval.copy()
# 1) Best model metrics bar (Accuracy, Precision, Recall, F1)
best = df_eval.sort_values("F1_Score", ascending=False).iloc[0]
labels = ["Accuracy","Precision","Recall","F1_Score"]
vals = [best.get("Accuracy",np.nan), best.get("Precision",np.nan), best.get("Recall",np.nan), best["F1_Score"]]
plt.bar(range(len(labels)), vals)
plt.xticks(range(len(labels)), labels, rotation=0)
plt.ylabel("Score"); plt.title(f"Best Model Metrics — {best['model']}")
for i,v in enumerate(vals): 
    if pd.notna(v): plt.text(i, v+0.003, f"{v:.3f}", ha="center", fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(OUT,"eval_best_model_bars.png"), dpi=200); plt.close()

# 2) Precision–Recall scatter for all models (size ~ F1)
prec_ok = df_eval["Precision"].notna() & df_eval["Recall"].notna()
d = df_eval[prec_ok].copy()
sizes = 300 * (d["F1_Score"] - d["F1_Score"].min()) / (d["F1_Score"].max() - d["F1_Score"].min() + 1e-9) + 20
plt.figure()
plt.scatter(d["Precision"], d["Recall"], s=sizes, alpha=0.7)
plt.xlabel("Precision"); plt.ylabel("Recall"); plt.title("Models: Precision vs Recall (size ~ F1)")
# annotate top3 + worst2 by F1
d_sorted = d.sort_values("F1_Score", ascending=False)
for _, r in pd.concat([d_sorted.head(3), d_sorted.tail(2)]).iterrows():
    plt.annotate(r["model"], (r["Precision"], r["Recall"]), xytext=(5,5), textcoords="offset points", fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(OUT,"eval_precision_recall_scatter.png"), dpi=200); plt.close()

# 3) F1 distribution histogram (quick overview)
plt.hist(df_eval["F1_Score"].dropna(), bins=12)
plt.xlabel("F1-Score"); plt.ylabel("Count"); plt.title("Distribution of F1 across Models")
plt.tight_layout(); plt.savefig(os.path.join(OUT,"eval_f1_hist.png"), dpi=200); plt.close()
