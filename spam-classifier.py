from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from sklearn.pipeline import Pipeline

import joblib   


import pandas as pd

data = pd.read_csv(
    "data/SMSSpamCollection.txt",
    sep="\t",
    header=None,
    names=["label", "message"]
)

# print(data.head())
# print(data.shape)
# print(data.columns)
# print(data["label"].value_counts())
# print(data.isnull().sum())
# print(data.dtypes)
# print(data["label"].value_counts(normalize=True) * 100)
# print(data[data["label"] == "spam"]["message"].head(10))
# print(data[data["label"] == "ham"]["message"].head(10))\
X = data["message"]
y = data["label"]
# print("X:")
# print(X.head())
# print("\ny:")
# print(y.head())
# print(X.shape)
# print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
# print("X_train:", X_train.shape)
# print("X_test:", X_test.shape)

# print("y_train:", y_train.shape)
# print("y_test:", y_test.shape)


vectorizer = TfidfVectorizer(min_df=2)  # we can modify here

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("Training shape:", X_train_tfidf.shape)
print("Testing shape:", X_test_tfidf.shape)

# model = MultinomialNB()
model = LogisticRegression(C=2)

model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

# print(y_pred[:20])


accuracy = accuracy_score(y_test, y_pred)

# print("Accuracy:", accuracy)

# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))

# print("\nConfusion Matrix:")
# print(confusion_matrix(y_test, y_pred))

## Test the model with new message 

# message = input("Enter a message: ")

# message_tfidf = vectorizer.transform([message])

# prediction = model.predict(message_tfidf)[0]

# if prediction == "spam":
#     print("🚨 This message is SPAM!")
# else:
#     print("✅ This message is NOT SPAM.")


# Logistick regression


logistic_model = LogisticRegression()

logistic_model.fit(X_train_tfidf, y_train)

logistic_pred = logistic_model.predict(X_test_tfidf)

# print(logistic_pred[:20])

#


logistic_accuracy = accuracy_score(y_test, logistic_pred)

print("Logistic Regression Accuracy:", logistic_accuracy)

# print("\nClassification Report:")
print(classification_report(y_test, logistic_pred))


# message = input("Enter a message: ")

# message_tfidf = vectorizer.transform([message])

# prediction = logistic_model.predict(message_tfidf)[0]

# if prediction == "spam":
#     print("🚨 This message is SPAM!")
# else:
#     print("✅ This message is NOT SPAM.")


# features = vectorizer.get_feature_names_out()

# print("Number of features:", len(features))
# print("First 50 features:")
# print(features[:50])


message = ["free prize"]

message_tfidf = vectorizer.transform(message)

# print(message_tfidf)
# print("Shape:", message_tfidf.shape)


# feature_names = vectorizer.get_feature_names_out()

# for index in message_tfidf.indices:
#     print(index, feature_names[index], message_tfidf[0, index])


#piippeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("model", LogisticRegression())
])
pipeline.fit(X_train, y_train)
pipeline_pred = pipeline.predict(X_test)


##
# from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


cm = confusion_matrix(y_test, logistic_pred)

# print("Confusion Matrix:")
# print(cm)

# display = ConfusionMatrixDisplay(
#     confusion_matrix=cm,
#     display_labels=["Ham", "Spam"]
# )

# display.plot()
# plt.title("Spam Classifier Confusion Matrix")
# plt.show()

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(min_df=2)),
    ("model", LogisticRegression(C=2))
])

pipeline.fit(X_train, y_train)

joblib.dump(pipeline, "spam_classifier.pkl")

print("Final model saved successfully!")

