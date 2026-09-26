# 📰 Fake News Detection System

A machine-learning based web application that analyzes news article text and predicts whether it is **possibly fake** or **possibly real**.

The project uses **TF-IDF text vectorization** and **Logistic Regression** to classify news articles.

> ⚠️ **Important:** This project provides a machine-learning prediction based on patterns learned from the training dataset. It does **not** independently fact-check or verify the truth of an article.

---

## 📌 Project Overview

Fake news can spread quickly through online platforms and social media. This project demonstrates how Natural Language Processing (NLP) and Machine Learning can be used to identify patterns associated with fake and real news articles.

Users can paste a news article into the web application and receive:

* 🔍 Predicted classification
* 📊 Model confidence
* ⚠️ Fake probability
* ✅ Real probability

---

## 🎯 Objectives

* Detect patterns in fake and real news articles.
* Apply Natural Language Processing techniques to text.
* Convert text into numerical features using TF-IDF.
* Train a Logistic Regression classification model.
* Build an interactive Streamlit web application.
* Provide prediction probabilities to the user.

---

## 🤖 Machine Learning Workflow

```text
News Dataset
     ↓
Data Preparation
     ↓
Text Cleaning / Filtering
     ↓
Train-Test Split
     ↓
TF-IDF Vectorization
     ↓
Logistic Regression
     ↓
Model Evaluation
     ↓
Streamlit Web Application
     ↓
Fake / Real Prediction
```

---

## 📊 Model Performance

The trained model achieved:

**98.75% test accuracy**

on the prepared dataset used for this project.

### Classification Report

| Class | Precision | Recall | F1-Score |
| ----- | --------- | ------ | -------- |
| FAKE  | 0.99      | 0.98   | 0.99     |
| REAL  | 0.98      | 0.99   | 0.99     |

The evaluation was performed on a held-out test set containing **8,854 articles**.

> The reported accuracy is a dataset-specific machine-learning evaluation and should not be interpreted as proof that the system can determine factual truth for arbitrary real-world news.

---

## 📂 Dataset

The project uses the **Fake and Real News Dataset**, containing articles labeled as:

* `FAKE`
* `REAL`

The original dataset files are intentionally not included in this GitHub repository because they are large.

The `.gitignore` file prevents the dataset files from being uploaded.

---

## 🛠️ Technologies Used

* Python
* Scikit-learn
* NumPy
* Streamlit
* Joblib
* TF-IDF
* Logistic Regression
* Natural Language Processing
* Git & GitHub

---

## 📁 Project Structure

```text
Fake-News-Detection/
│
├── data/
│   └── news.csv
│
├── models/
│   ├── fake_news_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── screenshots/
│
├── app.py
├── train.py
├── prepare_data.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Hemalatha-17032008/Fake-News-Detection.git
```

### 2. Open the project

```bash
cd Fake-News-Detection
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

### Windows

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

Paste a news article into the text box and click:

**🔍 Analyze News**

The application will display the predicted class and model probabilities.

---

## 🧠 Model Details

### TF-IDF

TF-IDF stands for **Term Frequency-Inverse Document Frequency**.

It converts text into numerical features based on how important words are within the collection of documents.

### Logistic Regression

Logistic Regression is used as the classification algorithm to predict whether an article belongs to the `FAKE` or `REAL` class.

---

## 🔮 Future Improvements

Possible future improvements include:

* Add more diverse and recent news datasets.
* Add text preprocessing and normalization.
* Compare multiple machine-learning algorithms.
* Add explainable AI features.
* Add a news source verification system.
* Integrate external fact-checking sources.
* Deploy the application online.
* Add multilingual news detection.
* Improve robustness against unfamiliar writing styles.

---

## ⚠️ Limitations

This system should not be treated as an authoritative fact-checking tool.

The model learns statistical patterns from its training data. A prediction can therefore be incorrect, particularly when an article differs substantially from the training data or when factual context is required.

---

## 👩‍💻 Author

**Hemalatha-17032008**

GitHub:
https://github.com/Hemalatha-17032008

---

## ⭐ Project

If you find this project useful for learning about Machine Learning and NLP, consider giving the repository a ⭐.
