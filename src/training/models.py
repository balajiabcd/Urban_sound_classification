from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def get_model_known_names():
    """
    Return a set/list of known model short-names.
    """
    return ["rf", "random_forest", "svc", "svm", "logreg", "knn"]

def get_model(name: str):
    """
    Construct a sklearn model by short-name.
    """
    name = (name or "").lower()
    if name in {"rf", "random_forest"}:
        return RandomForestClassifier(n_estimators=100, random_state=0)
    if name in {"svc", "svm"}:
        return SVC(probability=False, random_state=0)
    if name == "logreg":
        return LogisticRegression(max_iter=1000, n_jobs=None)
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=5)
    # default
    return RandomForestClassifier(n_estimators=100, random_state=0)
