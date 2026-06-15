from flask import Flask, request, jsonify, render_template
import joblib
import nltk
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from nltk.corpus import stopwords
from textblob import TextBlob
import language_tool_python
import spacy
import os
import docx
import PyPDF2

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

app = Flask(__name__)
model = joblib.load('essay_model.pkl')
stop_words = set(stopwords.words('english'))

print('Loading grammar tool...')
grammar_tool = language_tool_python.LanguageTool('en-US')
print('Loading spaCy...')
nlp = spacy.load('en_core_web_sm')
print('All tools loaded!')

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}

TRANSITION_WORDS = [
    'however', 'furthermore', 'therefore', 'moreover', 'consequently',
    'nevertheless', 'in conclusion', 'in addition', 'on the other hand',
    'for example', 'for instance', 'in contrast', 'as a result',
    'in summary', 'to conclude', 'firstly', 'secondly', 'finally',
    'additionally', 'similarly', 'meanwhile', 'subsequently'
]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_file(file):
    ext = file.filename.rsplit('.', 1)[1].lower()
    if ext == 'txt':
        return file.read().decode('utf-8')
    elif ext == 'pdf':
        reader = PyPDF2.PdfReader(file)
        return ''.join([page.extract_text() or '' for page in reader.pages])
    elif ext == 'docx':
        doc = docx.Document(file)
        return '\n'.join([para.text for para in doc.paragraphs])
    return ''

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

    return [[word_count, sentence_count, unique_words,
             round(avg_word_length, 3),
             round(vocab_richness, 3),
             round(avg_sentence_length, 3),
             round(stopword_ratio, 3),
             char_count,
             paragraph_count,
             round(pos_diversity, 3),
             round(noun_ratio, 3),
             round(verb_ratio, 3),
             round(adj_ratio, 3),
             round(sentiment_polarity, 3),
             grammar_error_count,
             named_entity_count,
             punctuation_count,
             comma_count,
             transition_count,
             round(grammar_error_rate, 4),
             round(sentence_length_variance, 3)]]

def get_feedback(score, word_count, unique_words, avg_sentence_length,
                 grammar_errors, named_entities, transition_count, grammar_error_rate):
    tips = []

    if score >= 8:
        tips.append("Excellent writing! Your essay is well structured and articulate.")
    elif score >= 6:
        tips.append("Good essay! A few improvements will make it great.")
    elif score >= 4:
        tips.append("Decent effort. Focus on developing your arguments more.")
    else:
        tips.append("Needs improvement. Try expanding your ideas with more detail.")

    if word_count < 150:
        tips.append("Try to write more — aim for at least 200 words.")
    elif word_count > 400:
        tips.append("Great length! Your essay is well developed.")

    vocab_richness = unique_words / (word_count ** 0.5) if word_count > 0 else 0
    if vocab_richness < 6:
        tips.append("Try using more varied vocabulary to strengthen your writing.")
    elif vocab_richness > 10:
        tips.append("Excellent vocabulary variety!")

    if avg_sentence_length < 8:
        tips.append("Your sentences are quite short. Try combining ideas for better flow.")
    elif avg_sentence_length > 25:
        tips.append("Some sentences are very long. Break them up for clarity.")

    if grammar_error_rate > 0.1:
        tips.append(f"Found {grammar_errors} grammar issues — proofread carefully.")
    elif grammar_errors == 0:
        tips.append("No grammar errors detected. Great job!")

    if named_entities > 5:
        tips.append("Good use of specific references and named entities.")

    if transition_count == 0:
        tips.append("Try using transition words (e.g. 'however', 'furthermore') to improve flow.")
    elif transition_count >= 3:
        tips.append("Good use of transition words — your essay flows well.")

    return " ".join(tips)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/score', methods=['POST'])
def score():
    essay = ''

    # Handle file upload
    if 'file' in request.files and request.files['file'].filename != '':
        file = request.files['file']
        if allowed_file(file.filename):
            essay = extract_text_from_file(file)
        else:
            return jsonify({'error': 'Unsupported file type. Use .txt, .pdf, or .docx'})

    # Handle JSON text (typed essay)
    elif request.is_json:
        essay = request.get_json().get('essay', '').strip()

    # Handle form text (fallback)
    elif request.form.get('essay'):
        essay = request.form.get('essay', '').strip()

    if not essay.strip():
        return jsonify({'error': 'No essay content found.'})

    if len(essay.split()) < 20:
        return jsonify({'error': 'Essay too short. Please write at least 20 words.'})

    features = extract_features(essay)
    raw_score = model.predict(features)[0]

    # Scores are already normalized to 0-10 from training
    score_out_of_10 = round(float(raw_score), 1)
    score_out_of_10 = min(10.0, max(0.0, score_out_of_10))

    # Stats for response
    words = [w for w in essay.split() if w.isalpha()]
    word_count = len(words)
    unique_words = len(set([w.lower() for w in words]))
    sentences = nltk.sent_tokenize(essay)
    avg_sentence_length = round(word_count / len(sentences), 1) if sentences else 0
    grammar_errors = len(grammar_tool.check(essay))
    named_entities = len(nlp(essay).ents)
    text_lower = essay.lower()
    transition_count = sum(1 for t in TRANSITION_WORDS if t in text_lower)
    grammar_error_rate = grammar_errors / word_count if word_count > 0 else 0

    if score_out_of_10 >= 8:
        grade = 'A'
    elif score_out_of_10 >= 6:
        grade = 'B'
    elif score_out_of_10 >= 4:
        grade = 'C'
    else:
        grade = 'D'

    return jsonify({
        'overall_score': score_out_of_10,
        'grade': grade,
        'word_count': word_count,
        'unique_words': unique_words,
        'sentence_count': len(sentences),
        'avg_sentence_length': avg_sentence_length,
        'grammar_errors': grammar_errors,
        'named_entities': named_entities,
        'transition_count': transition_count,
        'feedback': get_feedback(score_out_of_10, word_count, unique_words,
                                  avg_sentence_length, grammar_errors, named_entities,
                                  transition_count, grammar_error_rate)
    })

if __name__ == '__main__':
    app.run(debug=True)