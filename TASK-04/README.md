# 📰 Fake News Detection System

A machine learning system that classifies news headlines/articles as **REAL** or **FAKE**,
built with classic NLP (TF-IDF) + classical ML (Logistic Regression, Random Forest, Naive Bayes).

## Results

| Model | Test Accuracy |
|---|---|
| **Logistic Regression** ⭐ (best) | **92.5%** |
| Random Forest | 92.1% |
| Naive Bayes | 88.7% |

Trained and evaluated on ~4,600 labeled real-world news articles (balanced 50/50 REAL/FAKE),
20% held out as a test set. Full details, charts, and confusion matrices are in the notebook.

## Project structure

```
fake_news_project/
├── data/
│   └── news.csv                  # labeled dataset (title, text, label)
├── model/                        # created after training
│   ├── model.pkl                 # best trained classifier
│   ├── vectorizer.pkl            # fitted TF-IDF vectorizer
│   └── metadata.json             # which model won + all accuracies
├── preprocess.py                 # shared text-cleaning function
├── train_model.py                # data -> preprocess -> train -> evaluate -> save
├── Fake_News_Detection.ipynb     # full walkthrough notebook (already executed)
├── app.py                        # Streamlit interface — paste text, get a prediction
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

(First run will also auto-download a couple of small NLTK resources — punkt tokenizer
and the English stopword list.)

## 1. Train the model

Either run the script:

```bash
python3 train_model.py
```

...or open and run `Fake_News_Detection.ipynb` in Jupyter (it's already been executed once,
so you can also just read through it — data exploration, cleaning, TF-IDF, all three models
trained and compared, confusion matrices, and the save step). Both produce the same
`model/model.pkl` and `model/vectorizer.pkl`.

## 2. Launch the interface

```bash
streamlit run app.py
```

This opens a browser tab with a text box — paste in a headline or article and click **Check**
to get an instant REAL/FAKE prediction with confidence bars.

## How it works

1. **Data**: `title + text` from each article is combined into one field.
2. **Preprocessing** (`preprocess.py`): lowercase → strip URLs/HTML/punctuation → tokenize
   (NLTK) → remove stop words. The app applies this *exact same* function to whatever a user
   pastes in, so live input is cleaned identically to the training data.
3. **Features**: TF-IDF over unigrams + bigrams, top 5,000 terms.
4. **Models**: Logistic Regression, Random Forest, and Multinomial Naive Bayes are all trained
   and compared; the most accurate one is saved automatically.
5. **Interface**: `app.py` loads the saved model + vectorizer and serves predictions live.

## Dataset

[`GeorgeMcIntire/fake_real_news_dataset`](https://github.com/GeorgeMcIntire/fake_real_news_dataset)
— ~4,600 articles, REAL examples from mainstream outlets (e.g. Reuters), FAKE examples from
known fake-news sites, originally compiled for the widely-used tutorial
["How to Build a 'Fake News' Classification Model"](https://opendatascience.com/how-to-build-a-fake-news-classification-model/).

## ⚠️ Limitations — please read

This is a **student/portfolio-style demonstration**, not a production fact-checker:

- It learned *lexical and stylistic* patterns (word choice, phrasing) from ~4,600 articles
  collected mostly around 2016–2017 — it has no access to real-world facts, and doesn't verify
  claims against any source.
- It can be **confidently wrong** on topics, publications, or writing styles that differ from
  its training data (e.g. it misclassifies some genuine, dry Reuters-style economic headlines
  as FAKE simply because that topic/style is underrepresented in training).
- Treat predictions as a rough signal at best — always verify important claims against trusted,
  primary sources.

**Ideas to make it more robust:** a larger and more recent/diverse training set, source and
metadata features (publisher, author, date), word embeddings or a transformer model (e.g.
DistilBERT), and ongoing recalibration as misinformation tactics evolve.
