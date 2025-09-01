from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def build_candidates(seed=42):
    models = {}
    # KNN sweeps
    for metric in ["euclidean","manhattan","chebyshev","cosine","hamming"]:
        for k in range(1,10):
            name = f"knn_{metric}_k{k}"
            models[name] = KNeighborsClassifier(n_neighbors=k, metric=metric)
    # Tree / LR / RF
    models["tree_gini_d4"] = DecisionTreeClassifier(criterion="gini", max_depth=4, random_state=seed)
    models["logreg"] = LogisticRegression(solver="liblinear", max_iter=200)
    models["rf_400"] = RandomForestClassifier(n_estimators=400, random_state=seed, n_jobs=-1)
    models["rf_200"] = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    return models
