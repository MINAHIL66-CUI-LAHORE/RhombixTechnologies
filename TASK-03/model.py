import pandas as pd
import numpy as np
import nltk
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from nltk.corpus import stopwords
from textblob import TextBlob
import language_tool_python
import spacy
from tqdm import tqdm

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

stop_words = set(stopwords.words('english'))

print('Loading grammar tool...')
grammar_tool = language_tool_python.LanguageTool('en-US')
print('Loading spaCy...')
nlp = spacy.load('en_core_web_sm')
print('All tools loaded!')

# ---- Load Data ----
print("Loading dataset...")
df = pd.read_csv('training_set_rel3.tsv', sep='\t', encoding='latin-1')
df = df[['essay_set', 'essay', 'domain1_score']].dropna()
df.reset_index(drop=True, inplace=True)
print(f"Total essays in dataset: {len(df)}")
print(df['essay_set'].value_counts().sort_index())

# ---- ESSAY SET CONTROL ----
# Pick which sets to train on:
#
# OPTION A — Fast (2-3 hours): 4 sets
# SELECTED_SETS = [1, 2, 3, 4]
#
# OPTION B — Balanced (4-6 hours): 6 sets
# SELECTED_SETS = [1, 2, 3, 4, 5, 6]
#
# OPTION C — Full (6-12 hours, overnight): all 8 sets
SELECTED_SETS = [1, 2, 3, 4, 5, 6, 7, 8]
#
# OPTION D — Quick test only (~20-30 min): sample from 4 sets
# SELECTED_SETS = [1, 2, 3, 4]
# df = df[df['essay_set'].isin(SELECTED_SETS)].sample(500, random_state=42)
# ---- END ESSAY SET CONTROL ----

df = df[df['essay_set'].isin(SELECTED_SETS)]
df.reset_index(drop=True, inplace=True)
print(f"\nTraining on essay sets: {SELECTED_SETS}")
print(f"Essays per set:")
print(df['essay_set'].value_counts().sort_index())
print(f"Total: {len(df)} essays")

# Normalize scores per essay set to 0-10 scale
df['score_normalized'] = df.groupby('essay_set')['domain1_score'].transform(
    lambda x: (x - x.min()) / (x.max() - x.min()) * 10
)
print(f"Score range after normalization: {df['score_normalized'].min():.1f} - {df['score_normalized'].max():.1f}")

# ---- Feature Extraction ----
TRANSITION_WORDS = [
    'however', 'furthermore', 'therefore', 'moreover', 'consequently',
    'nevertheless', 'in conclusion', 'in addition', 'on the other hand',
    'for example', 'for instance', 'in contrast', 'as a result',
    'in summary', 'to conclude', 'firstly', 'secondly', 'finally',
    'additionally', 'similarly', 'meanwhile', 'subsequently'
]

def extract_features(text):
    sentences = nltk.sent_tokenize(text)
    words = nltk.word_tokenize(text.lower())
    words_only = [w for w in words if w.isalpha()]
    non_stop_words = [w for w in words_only if w not in stop_words]

    word_count = len(words_only)
    sentence_count = len(sentences)
    unique_words = len(set(words_only))

    avg_word_length = np.mean([len(w) for w in words_only]) if words_only else 0

    # Root TTR — does not penalize longer essays
    vocab_richness = unique_words / (word_count ** 0.5) if word_count > 0 else 0

    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
    stopword_ratio = len(non_stop_words) / word_count if word_count > 0 else 0
    char_count = len(text)

    # Paragraph count
    paragraph_count = len([p for p in text.split('\n') if p.strip()])

    # POS tagging
    pos_tags = nltk.pos_tag(words_only) if words_only else []
    pos_list = [tag for _, tag in pos_tags]
    total_pos = len(pos_list)
    unique_pos = len(set(pos_list))
    pos_diversity = unique_pos / total_pos if total_pos > 0 else 0
    noun_ratio = sum(1 for t in pos_list if t.startswith('NN')) / total_pos if total_pos > 0 else 0
    verb_ratio = sum(1 for t in pos_list if t.startswith('VB')) / total_pos if total_pos > 0 else 0
    adj_ratio  = sum(1 for t in pos_list if t.startswith('JJ')) / total_pos if total_pos > 0 else 0

    # Sentiment
    sentiment_polarity = TextBlob(text).sentiment.polarity

    # Grammar errors
    grammar_error_count = len(grammar_tool.check(text))

    # Named entities
    named_entity_count = len(nlp(text).ents)

    # Punctuation & structure
    punctuation_count = sum(1 for c in text if c in '.,;:!?')
    comma_count = text.count(',')

    # Transition words
    text_lower = text.lower()
    transition_count = sum(1 for t in TRANSITION_WORDS if t in text_lower)

    # Grammar error rate (normalized by word count)
    grammar_error_rate = grammar_error_count / word_count if word_count > 0 else 0

    # Sentence length variance
    sent_lengths = [len(nltk.word_tokenize(s)) for s in sentences]
    sentence_length_variance = np.var(sent_lengths) if sent_lengths else 0

    return {
        'word_count': word_count,
        'sentence_count': sentence_count,
        'unique_words': unique_words,
        'avg_word_length': round(avg_word_length, 3),
        'vocab_richness': round(vocab_richness, 3),
        'avg_sentence_length': round(avg_sentence_length, 3),
        'stopword_ratio': round(stopword_ratio, 3),
        'char_count': char_count,
        'paragraph_count': paragraph_count,
        'pos_diversity': round(pos_diversity, 3),
        'noun_ratio': round(noun_ratio, 3),
        'verb_ratio': round(verb_ratio, 3),
        'adj_ratio': round(adj_ratio, 3),
        'sentiment_polarity': round(sentiment_polarity, 3),
        'grammar_error_count': grammar_error_count,
        'named_entity_count': named_entity_count,
        'punctuation_count': punctuation_count,
        'comma_count': comma_count,
        'transition_count': transition_count,
        'grammar_error_rate': round(grammar_error_rate, 4),
        'sentence_length_variance': round(sentence_length_variance, 3),
    }

# ---- Apply Features with Progress Bar ----
print("\nExtracting features (this is the slow part)...")
tqdm.pandas(desc="Processing essays")
features = df['essay'].progress_apply(extract_features)
X = pd.DataFrame(features.tolist())
y = df['score_normalized']
print(f"\nFeatures extracted! Shape: {X.shape}")
print(f"Features ({len(X.columns)}): {list(X.columns)}")

# ---- Train Model ----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")

print("Training GradientBoosting model...")
model = GradientBoostingRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42
)
model.fit(X_train, y_train)

# ---- Evaluate ----
y_pred = model.predict(X_test)
y_pred = np.clip(y_pred, 0, 10)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("=" * 40)
print(f"Essay sets used: {SELECTED_SETS}")
print(f"Total essays:    {len(df)}")
print(f"RMSE:            {rmse:.3f}")
print(f"R2 Score:        {r2:.3f}")
print("=" * 40)

# Feature importance
importances = pd.Series(model.feature_importances_, index=X.columns)
print("\nTop 10 Most Important Features:")
print(importances.sort_values(ascending=False).head(10).to_string())

# ---- Save Model ----
joblib.dump(model, 'essay_model.pkl')
print("\nModel saved as essay_model.pkl — ready for Flask!")