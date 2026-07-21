"""
Shared text-preprocessing utilities for the Fake News Detection System.

Both the training script (train_model.py) and the interface (app.py) import
`clean_text` from here, so a headline typed into the app is cleaned in
*exactly* the same way as the training data. If the two ever drifted apart,
the model would see input that doesn't match what it was trained on.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Make sure the required NLTK corpora are available. quiet=True keeps this
# silent on repeat runs (it no-ops once the data is already downloaded).
for pkg, path in [("punkt", "tokenizers/punkt"),
                   ("punkt_tab", "tokenizers/punkt_tab"),
                   ("stopwords", "corpora/stopwords")]:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(pkg, quiet=True)

STOPWORDS = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """Lowercase, strip noise, tokenize, and remove stopwords.

    Steps (matching the brief: tokenization + stop word removal, ahead of
    TF-IDF vectorization):
      1. Lowercase everything.
      2. Strip URLs and HTML tags.
      3. Keep only letters (drop punctuation/digits/symbols).
      4. Tokenize with NLTK's word tokenizer.
      5. Drop English stopwords and very short tokens (<=2 chars).
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)   # URLs
    text = re.sub(r"<.*?>", " ", text)               # HTML tags
    text = re.sub(r"[^a-z\s]", " ", text)            # punctuation & digits
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    return " ".join(tokens)
