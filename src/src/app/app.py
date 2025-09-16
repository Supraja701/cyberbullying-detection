from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

clf = joblib.load("models/clf_nb.pkl")
cv = joblib.load("models/vectorizer.pkl")

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        message = request.form["text"]
        vect = cv.transform([message])
        prediction = clf.predict(vect)[0]
        return render_template("result.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
