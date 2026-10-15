import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

print("Loading datasets...")

fake = pd.read_csv("fake.csv")
real = pd.read_csv("true.csv")

fake.columns = fake.columns.str.strip().str.lower()
real.columns = real.columns.str.strip().str.lower()

def prepare_text(df):
    parts = []

    if "title" in df.columns:
        parts.append(df["title"].fillna("").astype(str))

    if "text" in df.columns:
        parts.append(df["text"].fillna("").astype(str))

    if not parts:
        raise ValueError(
            "Dataset must contain a 'text' or 'title' column."
        )

    combined = parts[0]

    for part in parts[1:]:
        combined = combined + " " + part

    return combined.str.strip()


fake_data = pd.DataFrame({
    "news": prepare_text(fake),
    "label": 0
})

real_data = pd.DataFrame({
    "news": prepare_text(real),
    "label": 1
})

print("\nChecking dataset labels...")

print("\nFake dataset sample:")
print(fake_data["news"].iloc[0][:300])
print("Label:", fake_data["label"].iloc[0])

print("\nReal dataset sample:")
print(real_data["news"].iloc[0][:300])
print("Label:", real_data["label"].iloc[0])

data = pd.concat(
    [fake_data, real_data],
    ignore_index=True
)

data["news"] = data["news"].fillna("").astype(str)
data = data[data["news"].str.strip() != ""]

data.drop_duplicates(subset=["news"], inplace=True)

data = data.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

print("\nDataset loaded successfully!")
print("Total articles:", len(data))
print("Fake articles:", (data["label"] == 0).sum())
print("Real articles:", (data["label"] == 1).sum())

X = data["news"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=100000,
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            C=1.0
        )
    )
])

print("\nTraining model...")

model.fit(X_train, y_train)

print("Training completed!")

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nTest Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["FAKE NEWS", "REAL NEWS"],
        zero_division=0
    )
)

def detect_news(news):
    news = news.strip()

    if not news:
        print("Please enter actual news text.")
        return

    probabilities = model.predict_proba([news])[0]

    fake_score = probabilities[0] * 100
    real_score = probabilities[1] * 100

    prediction = model.predict([news])[0]

    print("\n" + "-" * 40)

    if prediction == 1:
        print("PREDICTION: REAL NEWS")
    else:
        print("PREDICTION: FAKE NEWS")

    print(f"Fake score: {fake_score:.2f}%")
    print(f"Real score: {real_score:.2f}%")

    if max(fake_score, real_score) < 60:
        print("Warning: Model confidence is relatively low.")

    print("Note: Prediction is not independent fact verification.")
    print("-" * 40)

print("\n" + "=" * 45)
print("       FAKE NEWS DETECTION SYSTEM")
print("=" * 45)

print("Enter an article or type 'exit' to quit.")

while True:
    try:
        news = input("\nEnter news: ")

        if news.strip().lower() == "exit":
            print("Thank you!")
            break

        detect_news(news)

    except (KeyboardInterrupt, EOFError):
        print("\nProgram stopped.")
        break
