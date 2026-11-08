import joblib

model = joblib.load("spam_classifier.pkl")

message = input("Enter a message: ")

prediction = model.predict([message])[0]

probabilities = model.predict_proba([message])[0]

print("Classes:", model.classes_)
print("Prediction:", prediction)
print("Probabilities:", probabilities)

spam_probability = probabilities[1] * 100

print(f"Spam probability: {spam_probability:.2f}%")