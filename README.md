# 📰 Fake News Detection System

## 📌 About the Project

The **Fake News Detection System** is a machine learning-based Python project that classifies news articles as **Fake News** or **Real News**.

The project uses Natural Language Processing (NLP) techniques and a Logistic Regression model to analyze news text and predict its category based on patterns learned from a labeled dataset.

This project aims to demonstrate how machine learning can be applied to text classification and news analysis.

## 🎯 Objectives

* To detect potentially fake and real news articles using machine learning.
* To understand Natural Language Processing (NLP).
* To implement text classification using TF-IDF and Logistic Regression.
* To evaluate a machine learning model using standard evaluation metrics.

## ✨ Features

* 📰 Fake and real news classification.
* 🤖 Machine learning-based predictions.
* 🧠 Text processing using TF-IDF.
* 📊 Model evaluation using accuracy and classification metrics.
* 🐍 Simple Python implementation.

## 🛠️ Technologies Used

* **Programming Language:** Python
* **Libraries:** Pandas, Scikit-learn
* **Machine Learning Algorithm:** Logistic Regression
* **Text Feature Extraction:** TF-IDF Vectorization
* **Dataset Format:** CSV

## 📂 Project Structure

```text
Fake-News-Detection-System/
│
├── app.py
├── Fake.csv
├── True.csv
└── README.md
```

**File descriptions:**

* `app.py` — Contains the Python code for data preprocessing, model training, evaluation, and news prediction.
* `Fake.csv` — Dataset containing fake news articles.
* `True.csv` — Dataset containing real news articles.
* `README.md` — Project documentation and setup instructions.

**Important:** The CSV files must be downloaded separately if they are not included in the repository.

## 📥 Dataset Download Instructions

The project uses the Fake and Real News Dataset, which contains separate CSV files for fake and real news articles.

### Step 1: Download the Dataset

Download the dataset from Kaggle:

**Dataset:** [Fake and Real News Dataset — Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

### Step 2: Extract the Files

After downloading the dataset, extract the ZIP file.

You should find these two files:

* `Fake.csv`
* `True.csv`

### Step 3: Place the CSV Files

Copy both files into the same folder as `app.py`.

Your project folder should look like this:

```text
Fake-News-Detection-System/
│
├── app.py
├── Fake.csv
├── True.csv
└── README.md
```

**Note:** The filenames must match exactly, including capitalization.

## ⚙️ Installation and Setup

### Step 1: Install Python

Install Python 3 if it is not already installed.

### Step 2: Clone the Repository

```bash
git clone <your-repository-url>
```

### Step 3: Navigate to the Project Folder

```bash
cd Fake-News-Detection-System
```

### Step 4: Install Required Libraries

Open your terminal and run:

```bash
pip install pandas scikit-learn
```

### Step 5: Download the Dataset

Download the dataset from the Kaggle link above and place `Fake.csv` and `True.csv` in the project folder.

### Step 6: Run the Project

Execute the following command:

```bash
python app.py
```

Follow the instructions displayed in the terminal to enter a news article and view the prediction.

## 🔍 How It Works

1. **Dataset Loading:** The system loads fake and real news articles from the CSV files.
2. **Data Preprocessing:** It prepares the text data for machine learning.
3. **Feature Extraction:** TF-IDF converts the text into numerical features.
4. **Model Training:** Logistic Regression learns patterns from the labeled training data.
5. **Model Evaluation:** The model's performance is evaluated using metrics such as accuracy, precision, recall, and F1-score.
6. **Prediction:** The trained model predicts whether the entered article resembles fake or real news from its training data.

## 📊 Machine Learning Concepts Used

* Natural Language Processing (NLP)
* TF-IDF Vectorization
* Logistic Regression
* Training and Testing Data
* Classification Metrics

## ⚠️ Limitations

This system predicts news categories based on patterns learned from its training dataset. It does not independently verify news against trusted sources or establish whether a claim is factually true.

Predictions may be incorrect, especially for unfamiliar topics or news written in a different style from the training data.

## 🚀 Future Improvements

* Add a web-based user interface.
* Integrate trusted news sources for fact-checking.
* Improve model evaluation using additional datasets.
* Explore more advanced NLP models.

## 👨‍💻 Project Type

Machine Learning | Natural Language Processing | Text Classification

## 📄 License

This project is licensed under the MIT License, if the repository includes an MIT `LICENSE` file.
