import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

def train_model(data_path="data/processed/dataset.csv"):
    df = pd.read_csv(data_path)
    X = df["clean_tweet"]
    y = df["Text Label"]

    cv = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X = cv.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    joblib.dump(clf, "models/clf_nb.pkl")
    joblib.dump(cv, "models/vectorizer.pkl")
    print("Model and vectorizer saved in /models/")

if __name__ == "__main__":
    train_model()
