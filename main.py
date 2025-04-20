import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Download stopwords if not already downloaded
nltk.download('stopwords')

# Load the data
true_df = pd.read_csv("data/True.csv")
fake_df = pd.read_csv("data/Fake.csv")

# Add labels (0 = Real, 1 = Fake)
true_df['label'] = 0
fake_df['label'] = 1

# Combine both datasets
df = pd.concat([true_df, fake_df], ignore_index=True)

# Clean the text
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()  # lowercase
    text = ''.join([char for char in text if char not in string.punctuation])  # remove punctuation
    words = text.split()
    words = [word for word in words if word not in stop_words]  # remove stopwords
    return ' '.join(words)

df['cleaned_text'] = df['text'].apply(clean_text)

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_text'])  # Text to numbers
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print("\n✨ Model Accuracy:", accuracy)
from sklearn.metrics import classification_report, confusion_matrix

# Print classification report
print("\n📄 Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))

# Print confusion matrix
print("\n🧮 Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))
