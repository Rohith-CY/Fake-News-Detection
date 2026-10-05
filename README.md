# 📰 Fake News Detection

A machine learning-based web application that classifies news articles as **Fake News** or **Real News** using Natural Language Processing and machine learning.

## 📌 Overview

This project uses **TF-IDF** to convert news text into numerical features and compares multiple machine learning algorithms for classification.

The **Linear SVM** model achieved the best performance and is integrated into a **Flask** web application for real-time prediction.

## 🚀 Features

- Fake and real news classification
- TF-IDF text feature extraction
- Multiple machine learning models
- Linear SVM as the final model
- Flask-based web application
- Simple and responsive user interface
- Saved trained model and vectorizer

## 🧠 Machine Learning

The following models were evaluated:

- Multinomial Naive Bayes
- Logistic Regression
- Linear SVM

### Best Model

**Linear SVM**

| Metric | Score |
|--------|------:|
| Accuracy | 99.18% |
| Precision | 99.02% |
| Recall | 99.47% |
| F1-Score | 99.25% |

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF
- Linear SVM
- Flask
- HTML
- CSS
- Jupyter Notebook

## 📂 Project Structure

```text
Fake-News-Detection/
│
├── data/
│   ├── fake.csv
│   └── true.csv
│
├── models/
│   ├── model.pkl
│   └── vectorizer.pkl
│
├── notebooks/
│   └── Fake_news_detection.ipynb
│
├── templates/
│   └── index.html
│
├── app.py
├── README.md
└── .gitignore