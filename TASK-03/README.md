# Essay Scorer AI
## BY:MINAHIL AFTAB
## RHOMBIX TECHNOLOGIES
An AI-powered web application that automatically scores essays using NLP and Machine Learning.
![Essay Scorer Screenshot](assets/ESSAY-AI.png "Essay Scorer AI")
## Features
- Instant essay scoring out of 10
- Grade (A/B/C/D)
- Word count, unique words, sentence analysis
- AI feedback and writing tips

## Tech Stack
- Python, Flask, scikit-learn, NLTK
- Random Forest Regressor
- HTML, CSS, JavaScript
- Deployed on Render.com

## How to Run Locally

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add dataset
Place `training_set_rel3.tsv` in the root folder

### 3. Train the model
```bash
python model.py
```

### 4. Start the web app
```bash
python app.py
```

### 5. Open browser
```
http://localhost:5000
```

## Project Structure
```
essay_scorer/
├── Essay_Scoring.ipynb     # Full analysis notebook
├── model.py                # Train and save ML model
├── app.py                  # Flask backend
├── essay_model.pkl         # Saved trained model
├── requirements.txt
├── README.md
└── templates/
    └── index.html          # Frontend UI
```

## Dataset
ASAP Essay Scoring Dataset — 13,000 student essays with human scores
