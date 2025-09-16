import re, string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def clean_tweet(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)   # remove urls
    text = re.sub(r"@\w+", "", text)            # remove mentions
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

def preprocess_dataset(path="data/raw/dataset.csv", out_path="data/processed/dataset.csv"):
    df = pd.read_csv(path)
    df["clean_tweet"] = df["Tweet"].fillna("").apply(clean_tweet)
    df.to_csv(out_path, index=False)
    print(f"Processed data saved to {out_path}")

if __name__ == "__main__":
    preprocess_dataset()
