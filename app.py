from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Create a Flask app
app = Flask(__name__)


# Load the CSVs from your 'data' folder
true_df = pd.read_csv("data/True.csv")
fake_df = pd.read_csv("data/Fake.csv")

# Add a new column called 'label' (1 for real, 0 for fake)
true_df["label"] = 1
fake_df["label"] = 0

# Combine both into a single dataset
df = pd.concat([true_df, fake_df], ignore_index=True)



# Data Preprocessing
df['text'] = df['text'].fillna('')

# Feature Engineering
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(df['text'])
y = df['label']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle form submission and prediction
@app.route('/predict', methods=['POST'])
def predict():
    news_text = request.form['news']
    news_vectorized = vectorizer.transform([news_text])
    
    prediction = model.predict(news_vectorized)
    prediction_proba = model.predict_proba(news_vectorized)[0]

    result = 'Fake' if prediction[0] == 0 else 'Real'
    confidence = round(max(prediction_proba) * 100, 2)

    return render_template('index.html', result=f": {result}", confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)
