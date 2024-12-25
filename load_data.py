# load_data.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
file_path = r'C:\Users\admin\Downloads\archive\final_hateXplain.csv'
data = pd.read_csv(file_path)

# Define the features and labels
X = data['comment']
y = data['label']

# Encode the labels (normal, offensive, hatespeech)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Evaluate the model
y_pred = model.predict(X_test_tfidf)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'Precision: {precision_score(y_test, y_pred, average="weighted")}')
print(f'Recall: {recall_score(y_test, y_pred, average="weighted")}')
print(f'F1-Score: {f1_score(y_test, y_pred, average="weighted")}')
print(f'Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}')

# Save the model, vectorizer, and label encoder
joblib.dump(model, 'final_model.pkl')
joblib.dump(vectorizer, 'final_vectorizer.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
