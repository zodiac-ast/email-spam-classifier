
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

import joblib
import pandas as pd


# Load dataset
data = pd.read_csv(
    "data/SMSSpamCollection.txt",
    sep="\t",
    header=None,
    names=["label", "message"]
)

X = data["message"]
y = data["label"]


# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# TF-IDF Vectorization
vectorizer = TfidfVectorizer(min_df=2)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("Training shape:", X_train_tfidf.shape)
print("Testing shape:", X_test_tfidf.shape)


# Train Logistic Regression model
# model = MultinomialNB()
model = LogisticRegression(C=2)

model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)


# Test the model with a new message

# Logistic Regression
logistic_model = LogisticRegression()

logistic_model.fit(X_train_tfidf, y_train)

logistic_pred = logistic_model.predict(X_test_tfidf)

logistic_accuracy = accuracy_score(y_test, logistic_pred)

print("Logistic Regression Accuracy:", logistic_accuracy)

print(classification_report(y_test, logistic_pred))


# Test with a new message
message = ["free prize"]

message_tfidf = vectorizer.transform(message)


# Pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("model", LogisticRegression())
])

pipeline.fit(X_train, y_train)
pipeline_pred = pipeline.predict(X_test)


# Confusion Matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

cm = confusion_matrix(y_test, logistic_pred)


# Final pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(min_df=2)),
    ("model", LogisticRegression(C=2))
])

pipeline.fit(X_train, y_train)

joblib.dump(pipeline, "spam_classifier.pkl")

print("Final model saved successfully!")
