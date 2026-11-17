# 📧 Email Spam Classifier

A Machine Learning application that classifies text messages as **Spam** or **Ham (Not Spam)** using Natural Language Processing (NLP), TF-IDF, and Logistic Regression.

\<p align="center">   \<a href="[https://email-spam-classifier-sdpp6ufi8ofctyhavu6b2o.streamlit.app/](https://email-spam-classifier-sdpp6ufi8ofctyhavu6b2o.streamlit.app/)">     \<strong>🚀 Try the Live Application\</strong>   \</a>   |   \<a href="[https://github.com/zodiac-ast/email-spam-classifier](https://github.com/zodiac-ast/email-spam-classifier)">     \<strong>💻 View Source Code\</strong>   \</a> \</p>

## 📌 Project Overview

This project uses machine learning to identify whether a text message is spam or legitimate (ham).

The application accepts a message, processes it through a trained machine learning pipeline, and displays the predicted category and its associated probabilities.

## ✨ Features

- Spam and ham message classification.
- Text preprocessing using TF-IDF.
- Logistic Regression classification.
- Spam and ham probability display.
- Interactive Streamlit web interface.
- Saved machine learning pipeline using Joblib.

## 🧠 Machine Learning Approach

The project uses two main components:

1. **TF-IDF Vectorizer:** Converts text messages into numerical features that a machine learning model can process.
2. **Logistic Regression:** Learns patterns in the training messages and predicts whether a new message is spam or ham.

Both components are saved together in `spam_classifier.pkl` as a scikit-learn pipeline.

## 📊 Model Performance

Evaluation results on the test dataset:

| Metric         | Result |
| -------------- | -----: |
| Accuracy       | 97.76% |
| Spam Precision |   1.00 |
| Spam Recall    |   0.83 |
| Spam F1-score  |   0.91 |

These results describe performance on the test dataset and do not guarantee the same performance on new messages.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF
- Logistic Regression
- Streamlit
- Joblib

## 📂 Project Structure

```text
email-spam-classifier/
│
├── data/
│   └── SMSSpamCollection
│
├── app.py
├── spam-classifier.py
├── spam_classifier.pkl
├── test_model.py
├── requirements.txt
└── README.md
```

## ⚙️ Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/zodiac-ast/email-spam-classifier.git
```

### 2. Open the project folder

```bash
cd email-spam-classifier
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## 🚀 Live Demo

Try the deployed application:

[https://email-spam-classifier-sdpp6ufi8ofctyhavu6b2o.streamlit.app/](https://email-spam-classifier-sdpp6ufi8ofctyhavu6b2o.streamlit.app/)

## ⚠️ Limitations

- The model may incorrectly classify some messages.
- Its performance depends on the training data and the language and patterns of incoming messages.
- The displayed probabilities are model estimates, not guarantees.
- Important messages should always be reviewed rather than relying entirely on the prediction.

This project is an educational machine learning demonstration, not a guaranteed spam-filtering service.

## 👨‍💻 Author

**Machine Learning Project — Email Spam Classification**

GitHub: [https://github.com/zodiac-ast](https://github.com/zodiac-ast)
