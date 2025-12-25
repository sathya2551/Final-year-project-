import json
import nltk
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

lem = WordNetLemmatizer()

def normalize(text):
    tokens = [t for t in word_tokenize(text.lower()) if t.isalnum()]
    return " ".join(lem.lemmatize(t) for t in tokens)

with open("intents.json", "r", encoding="utf-8-sig") as file:
    data = json.load(file)

X = []
y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        X.append(normalize(pattern))
        y.append(intent["tag"])

vectorizer = TfidfVectorizer(ngram_range=(1,2))
X_vectorized = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X_vectorized, y)

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model trained successfully")
