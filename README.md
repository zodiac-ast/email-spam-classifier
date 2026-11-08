# 📧 Email Spam Classifier

A Machine Learning application that classifies text messages as **Spam** or **Ham (Not Spam)** using Natural Language Processing (NLP).

## 🚀 Project Overview

This project uses:
- **TF-IDF Vectorization** to convert text messages into numerical features.
- **Logistic Regression** to classify messages.
- **Streamlit** to build an interactive web application.
- **Joblib** to save and load the trained ML pipeline.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib

## 📂 Project Structure

```text
email-spam-classifier/
├── app.py
├── spam-classifier.py
├── spam_classifier.pkl
├── test_model.py
├── requirements.txt
└── README.md
```

## ▶️ Run Locally

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start the application:

```bash
streamlit run app.py
```

3. Enter a message and click **Check Message**.

## 🎯 Features

- Classifies messages as Spam or Ham.
- Displays spam probability.
- Displays ham probability.
- Provides an interactive Streamlit interface.

## 👨‍💻 Author

Machine Learning Project — Email Spam Classification