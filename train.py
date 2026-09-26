import csv
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# -----------------------------
# 1. Load dataset
# -----------------------------

texts = []
labels = []

with open("data/news.csv", "r", encoding="utf-8", errors="ignore") as file:

    reader = csv.DictReader(file)

    for row in reader:

        text = row.get("text", "")
        label = row.get("label", "")

        if text.strip() and label.strip():

            texts.append(text)
            labels.append(label)


print("Dataset loaded successfully!")
print("Total articles:", len(texts))


# -----------------------------
# 2. Split dataset
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.20,
    random_state=42,
    stratify=labels
)


print("Training articles:", len(X_train))
print("Testing articles:", len(X_test))


# -----------------------------
# 3. Convert text to TF-IDF
# -----------------------------

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=50000
)

X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)


print("TF-IDF conversion completed!")


# -----------------------------
# 4. Create ML model
# -----------------------------

print("\nTraining Logistic Regression model...")

model = LogisticRegression(
    max_iter=1000
)


# -----------------------------
# 5. Train model
# -----------------------------

model.fit(
    X_train_tfidf,
    y_train
)


print("Model training completed!")


# -----------------------------
# 6. Make predictions
# -----------------------------

predictions = model.predict(X_test_tfidf)


# -----------------------------
# 7. Calculate accuracy
# -----------------------------

accuracy = accuracy_score(
    y_test,
    predictions
)


print("\n==============================")
print("MODEL RESULTS")
print("==============================")

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions
    )
)


# -----------------------------
# 8. Save model
# -----------------------------

joblib.dump(
    model,
    "models/fake_news_model.pkl"
)

joblib.dump(
    vectorizer,
    "models/tfidf_vectorizer.pkl"
)


print("\n==============================")
print("FILES SAVED")
print("==============================")

print("models/fake_news_model.pkl")
print("models/tfidf_vectorizer.pkl")

print("\nTraining completed successfully! 🚀")