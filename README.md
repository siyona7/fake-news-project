"# Fake News Detection"
# 📰 Fake News Detection Web Application

A lightweight yet powerful web application built to identify whether a given news article is *real* or *fake*, using machine learning techniques. Designed to help users combat misinformation in today’s digital era.

---

## 📌 Project Objective

With the rise in misinformation and digital propaganda, the goal of this project is to create a user-friendly tool that:
- Detects fake news in real time
- Demonstrates the practical use of machine learning in real-world problems
- Encourages awareness about media literacy

---

## ⚙️ Features

- 🔍 Text classification using **TF-IDF** & **Logistic Regression**
- 🧠 Trained on real-world news datasets (Kaggle: Fake.csv & True.csv)
- 📊 Confidence score visualization
- 🌙 Dark mode toggle
- 🗣️ Chatbot assistant for guiding users
- ⚡ Packaged executable for offline use

---

## 🧠 Technologies Used

| Domain | Tools / Libraries |
|--------|-------------------|
| Language | Python |
| ML Libraries | Scikit-learn, Pandas, NumPy |
| Frontend | HTML, CSS, JavaScript |
| Backend | Flask |
| Deployment | GitHub |

---

## 🗂️ Project Structure
fake-news-project/
│
├── static/ # CSS, JS files, images
├── templates/ # HTML templates
├── data/ # Training datasets (Fake.csv, True.csv)
├── model/ # Saved ML model (Pickle file)
├── app.py # Main Flask backend
├── requirements.txt # Required Python packages
├── README.md # This file
└── dist/ # Packaged app (if applicable)

## 🚀 How to Run Locally

1. Clone the repository:
git clone https://github.com/siyona7/fake-news-project.git
cd fake-news-project

2. Install required packages:
pip install -r requirements.txt

3. Run the app:
python app.py

4. Open your browser and go to:
http://127.0.0.1:5000/

✅ Make sure `Fake.csv` and `True.csv` are placed in the `/data` folder.


