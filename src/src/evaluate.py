import pandas as pd
import joblib
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def evaluate(data_path="data/processed/dataset.csv"):
    df = pd.read_csv(data_path)
    X = df["clean_tweet"]
    y = df["Text Label"]

    cv = joblib.load("models/vectorizer.pkl")
    clf = joblib.load("models/clf_nb.pkl")

    X = cv.transform(X)
    y_pred = clf.predict(X)

    print("Accuracy:", accuracy_score(y, y_pred))
    print(classification_report(y, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y, y_pred))

if __name__ == "__main__":
    evaluate()
