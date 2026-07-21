"""
Fake News Detection System — interface.

Paste a headline or article and get an instant REAL / FAKE prediction with
confidence scores, using the model trained in Fake_News_Detection.ipynb /
train_model.py.

Run with:  python3 -m streamlit run app.py
"""

import json
from pathlib import Path

import joblib
import streamlit as st

from preprocess import clean_text

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"

st.set_page_config(page_title="Fake News Detector", page_icon="⚠️", layout="centered")


# ─────────────────────────────────────────────────────────────────────────────
# THEME — hazard console: black chassis, safety-yellow accents, stamped verdict
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Anton&family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

    :root{
        --bg:        #0A0A0C;
        --panel-1:   #17171B;
        --panel-2:   #1F1F25;
        --yellow:    #FFD400;
        --yellow-d:  #C9A600;
        --white:     #F5F5F0;
        --muted:     #8B8B93;
        --red:       #FF3B30;
        --green:     #2ECC71;
        --stripe:    repeating-linear-gradient(45deg, var(--yellow) 0px, var(--yellow) 18px, #0A0A0C 18px, #0A0A0C 36px);
    }
html, body, [class*="st-"], .stApp { background-color: var(--bg) !important; color: var(--white); }
body, .stApp { font-family:'Inter', sans-serif; }
    .block-container { padding-top: 1rem !important; max-width: 760px; }
    #MainMenu, footer, header { visibility: hidden; }

    /* ---------- hazard rule ---------- */
    .hazard-bar{ height:10px; width:100%; background:var(--stripe); border-radius:3px; margin-bottom:22px; box-shadow: 0 3px 0 rgba(0,0,0,.6);}
    .hazard-bar.small{ height:6px; margin: 28px 0 18px 0; opacity:.9; }

    /* ---------- eyebrow / phase tag ---------- */
    .eyebrow{
        display:inline-flex; align-items:center; gap:10px;
        font-family:'JetBrains Mono', monospace; font-weight:700; letter-spacing:.12em;
        font-size:0.78rem; color:var(--bg); background:var(--yellow);
        padding:5px 12px; border-radius:4px;
        box-shadow: 0 3px 0 var(--yellow-d);
        text-transform:uppercase;
    }
    .eyebrow span.dim{ color:#0A0A0C; opacity:.55; font-weight:500;}

    /* ---------- hero ---------- */
    .hero-title{
        font-family:'Anton', sans-serif; font-weight:400; text-transform:uppercase;
        font-size:3.35rem; line-height:0.96; letter-spacing:.005em; margin:16px 0 6px 0;
        color:var(--white);
    }
    .hero-title .hl{ color:var(--yellow); text-shadow: 3px 3px 0 rgba(0,0,0,.5); }
    .hero-sub{ color:var(--muted); font-size:1rem; line-height:1.55; max-width:560px; margin-bottom:6px; }

    /* ---------- console panel (holds the textarea) ---------- */
    .console-wrap{
        background: var(--panel-1);
        border: 2px solid #000;
        border-radius: 14px;
        padding: 22px 22px 14px 22px;
        margin-top: 22px;
        box-shadow:
            0 1px 0 rgba(255,255,255,.04) inset,
            0 14px 0 -6px #000,
            0 18px 26px rgba(0,0,0,.55);
        position: relative;
    }
    .console-wrap::before{
        content:"INPUT FEED";
        position:absolute; top:-11px; left:18px;
        background:var(--bg); color:var(--yellow);
        font-family:'JetBrains Mono',monospace; font-size:.7rem; letter-spacing:.15em;
        padding:0 8px; font-weight:700;
    }
    .console-wrap::after{
        content:"● REC"; position:absolute; top:-11px; right:18px;
        background:var(--bg); color:var(--red);
        font-family:'JetBrains Mono',monospace; font-size:.68rem; letter-spacing:.1em; font-weight:700;
        padding:0 8px;
    }

    /* textarea */
    .stTextArea textarea{
        background: var(--panel-2) !important; color: var(--white) !important;
        border: 1px solid #35353c !important; border-radius: 8px !important;
        font-family:'JetBrains Mono', monospace !important; font-size: 0.92rem !important;
    }
    .stTextArea textarea:focus{ border-color: var(--yellow) !important; box-shadow: 0 0 0 1px var(--yellow) !important; }
    .stTextArea textarea::placeholder{ color:#5c5c66 !important; }
    label[data-testid="stWidgetLabel"] p{ color: var(--muted) !important; font-family:'JetBrains Mono',monospace; font-size:.78rem; letter-spacing:.06em; text-transform:uppercase;}

    /* ---------- chunky 3D button ---------- */
    div[data-testid="stButton"] button{
        background: var(--yellow) !important;
        color: #0A0A0C !important;
        font-family:'Anton', sans-serif !important;
        letter-spacing:.04em;
        font-size: 1.05rem !important;
        text-transform: uppercase;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.7rem 1rem !important;
        box-shadow: 0 6px 0 var(--yellow-d), 0 10px 18px rgba(0,0,0,.5) !important;
        transition: transform .06s ease, box-shadow .06s ease !important;
        width: 100%;
    }
    div[data-testid="stButton"] button:hover{ filter:brightness(1.04); }
    div[data-testid="stButton"] button:active{
        transform: translateY(6px) !important;
        box-shadow: 0 0 0 var(--yellow-d), 0 3px 6px rgba(0,0,0,.5) !important;
    }

    /* ---------- verdict stamp (the signature element) ---------- */
    .verdict-row{ display:flex; align-items:center; gap:26px; margin: 30px 0 6px 0; }
    .stamp{
        flex: 0 0 auto;
        width: 128px; height: 128px; border-radius: 50%;
        display:flex; align-items:center; justify-content:center;
        border: 6px solid var(--yellow);
        background: radial-gradient(circle at 35% 30%, var(--panel-2), #0A0A0C 70%);
        box-shadow: 0 10px 0 -2px #000, 0 16px 24px rgba(0,0,0,.6), inset 0 0 18px rgba(0,0,0,.6);
        transform: rotate(-8deg);
        position: relative;
    }
    .stamp.fake{ border-color: var(--red); }
    .stamp.real{ border-color: var(--green); }
    .stamp-text{
        font-family:'Anton', sans-serif; font-size:1.55rem; letter-spacing:.03em;
        color: var(--yellow); text-align:center; line-height:1;
    }
    .stamp.fake .stamp-text{ color: var(--red); }
    .stamp.real .stamp-text{ color: var(--green); }
    .verdict-copy{ font-family:'JetBrains Mono', monospace; }
    .verdict-copy .big{ font-family:'Anton', sans-serif; font-size:1.5rem; letter-spacing:.02em; text-transform:uppercase; margin-bottom:4px;}
    .verdict-copy .big.fake{ color: var(--red); }
    .verdict-copy .big.real{ color: var(--green); }
    .verdict-copy .desc{ color: var(--muted); font-size:.85rem; }

    /* ---------- confidence bars ---------- */
    .conf-label{ font-family:'JetBrains Mono', monospace; font-size:.78rem; letter-spacing:.08em; color:var(--muted); text-transform:uppercase; margin: 18px 0 4px 0;}
    div[data-testid="stProgress"] > div > div{ background: var(--panel-2) !important; border-radius: 6px !important; height: 14px !important;}
    div[data-testid="stProgress"] > div > div > div{ border-radius: 6px !important; }
    div[data-testid="stProgress"]:nth-of-type(1) > div > div > div{ background: var(--green) !important; }
    div[data-testid="stProgress"]:nth-of-type(2) > div > div > div{ background: var(--red) !important; }

    /* alerts */
    div[data-testid="stAlert"]{ background: var(--panel-1) !important; border-radius: 8px !important; border: 1px solid #2a2a30 !important;}

    /* expander */
    details{ background: var(--panel-1) !important; border: 1px solid #2a2a30 !important; border-radius: 8px !important; }
summary p{ color: var(--yellow) !important; font-family:'JetBrains Mono',monospace !important; font-weight:700 !important; letter-spacing:.04em; }
    .footer-note{ color:#55555e; font-size:.75rem; font-family:'JetBrains Mono',monospace; margin-top:10px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_DIR / "model.pkl")
    vectorizer = joblib.load(MODEL_DIR / "vectorizer.pkl")
    meta = {}
    meta_path = MODEL_DIR / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    return model, vectorizer, meta


# ─────────────────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="hazard-bar"></div>', unsafe_allow_html=True)
st.markdown(
    '<span class="eyebrow">PHASE 01 <span class="dim">/ INPUT</span></span>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero-title">Spot the<br><span class="hl">fake.</span></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero-sub">Paste a headline or article below. The model reads it for the '
    'lexical and stylistic fingerprints of unreliable reporting, then stamps a verdict — '
    '<b style="color:var(--white)">REAL</b> or <b style="color:var(--red)">FAKE</b> — with a confidence readout.</div>',
    unsafe_allow_html=True,
)

if not (MODEL_DIR / "model.pkl").exists():
    st.error(
        "No trained model found in `model/`. Run `python3 train_model.py` "
        "(or execute `Fake_News_Detection.ipynb`) first to train and save one."
    )
    st.stop()

model, vectorizer, meta = load_artifacts()

# ─────────────────────────────────────────────────────────────────────────────
# CONSOLE / INPUT
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="console-wrap">', unsafe_allow_html=True)
text_input = st.text_area(
    "Article text",
    height=180,
    placeholder="Paste a headline or article here...",
    label_visibility="visible",
)
check_clicked = st.button("Analyze", type="primary", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# VERDICT
# ─────────────────────────────────────────────────────────────────────────────
if check_clicked:
    if not text_input.strip():
        st.warning("Please paste some text first.")
    else:
        cleaned = clean_text(text_input)
        if not cleaned:
            st.warning("Couldn't find any usable words in that text — try pasting more of the article.")
        else:
            vec = vectorizer.transform([cleaned])
            pred = model.predict(vec)[0]

            css_class = "fake" if pred == "FAKE" else "real"
            verdict_word = "FAKE" if pred == "FAKE" else "REAL"
            desc = (
                "Flagged — this reads like patterns seen in unreliable reporting."
                if pred == "FAKE"
                else "Cleared — this reads consistent with reliable reporting."
            )

            st.markdown('<div class="hazard-bar small"></div>', unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="verdict-row">
                    <div class="stamp {css_class}"><div class="stamp-text">{verdict_word}</div></div>
                    <div class="verdict-copy">
                        <div class="big {css_class}">Verdict: {verdict_word}</div>
                        <div class="desc">{desc}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if hasattr(model, "predict_proba"):
                proba = dict(zip(model.classes_, model.predict_proba(vec)[0]))
                st.markdown('<div class="conf-label">Confidence readout</div>', unsafe_allow_html=True)
                for cls in ["REAL", "FAKE"]:
                    p = float(proba.get(cls, 0))
                    st.progress(p, text=f"{cls} — {p * 100:.1f}%")

st.markdown('<div class="hazard-bar small"></div>', unsafe_allow_html=True)
with st.expander("About this model"):
    if meta:
        st.write(f"**Model type:** {meta.get('best_model', 'n/a')}")
        st.write("**Test-set accuracy by model:**")
        for name, acc in meta.get("accuracies", {}).items():
            st.write(f"- {name}: {acc:.1%}")
    st.caption(
        "Trained on ~4,600 labeled news articles (title + body) using TF-IDF "
        "features and classical ML — see `Fake_News_Detection.ipynb` for the "
        "full pipeline. This is a demonstration/educational model, not a "
        "fact-checking tool: it recognizes lexical and stylistic patterns "
        "from its training data and can be confidently wrong on topics, "
        "outlets, or writing styles outside that distribution. Always verify "
        "important claims against trusted, primary sources."
    )
st.markdown('<div class="footer-note">HAZARD-CLASS DEMO MODEL · NOT A FACT-CHECKER</div>', unsafe_allow_html=True)
