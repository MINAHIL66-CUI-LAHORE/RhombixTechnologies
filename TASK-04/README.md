# Fake News Detection System

A machine learning project that classifies news articles as **REAL** or **FAKE** using Natural Language Processing (NLP) and classical machine learning algorithms. The system preprocesses news text, extracts TF-IDF features, compares multiple machine learning models, automatically selects the best-performing classifier, and provides real-time predictions through a Streamlit web application.

---

## Model Performance

| Model | Test Accuracy |
| :--- | :---: |
| **Logistic Regression (Best)** | **92.5%** |
| Random Forest | 92.1% |
| Multinomial Naive Bayes | 88.7% |

The highest-performing model is automatically selected and saved for deployment.

---

# Application Screenshots

## Home Page

<p align="center">
  <img src="images/home.png" width="900">
</p>

---

## Prediction Result

<p align="center">
  <img src="images/prediction.png" width="900">
</p>

---

## Confidence Scores

<p align="center">
  <img src="images/confidence.png" width="900">
</p>

---

## About Model

<p align="center">
  <img src="images/about_model.png" width="900">
</p>

---

## Project Structure

```text
fake_news_project/
│
├── data/
│   ├── True.csv
│   └── Fake.csv
│
├── images/
│   ├── home.png
│   ├── prediction.png
│   ├── confidence.png
│   └── about_model.png
│
├── model/
│   ├── model.pkl
│   ├── vectorizer.pkl
│   └── metadata.json
│
├── preprocess.py
├── train_model.py
├── Fake_News_Detection.ipynb
├── app.py
├── requirements.txt
└── README.md
```

---

## Installation

Install all required dependencies:

```bash
pip install -r requirements.txt
```

During the first execution, NLTK automatically downloads the required resources:

- punkt
- stopwords

---

## Training the Model

Run the training script:

```bash
python train_model.py
```

or execute the notebook:

```
Fake_News_Detection.ipynb
```

The training pipeline:

- Loads `True.csv` and `Fake.csv`
- Assigns REAL and FAKE labels
- Combines article title and body
- Removes publisher-specific label leakage
- Cleans and preprocesses text
- Generates TF-IDF features
- Trains multiple machine learning models
- Evaluates each model
- Saves the best-performing model automatically

Generated files:

```text
model/
├── model.pkl
├── vectorizer.pkl
└── metadata.json
```

---

## Running the Application

Launch the Streamlit application:

```bash
streamlit run app.py
```

The application allows users to paste a news headline or article and instantly receive:

- REAL or FAKE prediction
- Prediction confidence scores
- Best-performing model information
- Accuracy of all trained models

---

## How It Works

### 1. Data Loading

The project loads two datasets:

- `True.csv`
- `Fake.csv`

REAL and FAKE labels are assigned before combining both datasets.

---

### 2. Source Leakage Removal

Many genuine articles begin with publisher datelines such as:

```
WASHINGTON (Reuters) -
```

These patterns unintentionally reveal the correct label.

To reduce dataset bias, the training pipeline removes:

- Reuters datelines
- AP datelines
- AFP datelines
- Mentions of "Reuters"

This encourages the model to learn writing patterns rather than publisher identities.

---

### 3. Text Preprocessing

The preprocessing pipeline performs:

- Convert text to lowercase
- Remove URLs
- Remove HTML tags
- Remove punctuation
- Tokenize text using NLTK
- Remove English stopwords

The same preprocessing function is used during both training and prediction.

---

### 4. Feature Extraction

The cleaned text is converted into numerical vectors using TF-IDF with:

- Maximum Features: **5,000**
- Unigrams and Bigrams
- Minimum Document Frequency: **2**

---

### 5. Machine Learning Models

Three classifiers are trained and compared:

- Logistic Regression
- Random Forest
- Multinomial Naive Bayes

The classifier with the highest accuracy is automatically selected and saved.

---

## Dataset

This project uses the **Fake and Real News Dataset** by **clmentbisaillon**.

The dataset contains two CSV files:

- `True.csv` — Genuine news articles
- `Fake.csv` — Fake news articles

Each article contains:

- Title
- Text
- Subject
- Date

The combined dataset is shuffled before splitting into training and testing sets to avoid label-order bias.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Joblib
- Streamlit
- JSON
- Jupyter Notebook

---

## Saved Artifacts

| File | Description |
|------|-------------|
| `model.pkl` | Best-performing trained classifier |
| `vectorizer.pkl` | Trained TF-IDF vectorizer |
| `metadata.json` | Best model name and evaluation results |

---

## Limitations

This project is intended for educational and portfolio purposes.

Although publisher-specific patterns are removed to reduce dataset bias, the model does **not** verify factual claims using external sources. Instead, it learns statistical language patterns from historical news articles.

Performance may decrease on:

- Recent news events
- New writing styles
- Unseen publishers
- Emerging topics
- Articles outside the training distribution

Predictions should be treated as probabilistic estimates rather than factual verification.

---

## Future Improvements

Possible enhancements include:

- Larger and more diverse datasets
- Transformer-based models (BERT, DistilBERT, RoBERTa)
- Explainable AI (LIME, SHAP)
- Publisher credibility features
- Author metadata
- Real-time fact-checking APIs
- Continuous model retraining
- News source reliability analysis

---

## Submitted By

**Minahil Aftab**
