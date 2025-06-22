import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from .utils import load_data

def train_and_serialize(data_csv:str, out_dir:str="src/models"):
    X_train, X_test, y_train, y_test, scaler = load_data(data_csv)

    models = {
        'logistic_regression': LogisticRegression(max_iter=1000),
        'svm': SVC(probability=True),
        'decision_tree': DecisionTreeClassifier(),
        'knn': KNeighborsClassifier()
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        joblib.dump(model, f"{out_dir}/{name}.pkl")
    joblib.dump(scaler, f"{out_dir}/scaler.pkl")
    print("Models and scaler serialitzats correctament.")

if __name__ == "__main__":
    import os
    os.makedirs("src/models", exist_ok=True)
    train_and_serialize("pulsar_stars.csv")

