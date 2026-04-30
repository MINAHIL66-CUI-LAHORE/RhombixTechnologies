import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.preprocessing import LabelEncoder

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Muse AI — Music Recommender",
    page_icon  = "🎵",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

# ─── SPOTIFY CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Circular+Std:wght@400;500;700;900&family=Figtree:wght@300;400;500;600;700;800;900&display=swap');

:root {
    --sp-black:      #000000;
    --sp-dark:       #121212;
    --sp-card:       #181818;
    --sp-card-hover: #282828;
    --sp-green:      #1DB954;
    --sp-green-dark: #1aa34a;
    --sp-green-dim:  rgba(29,185,84,0.12);
    --sp-text:       #FFFFFF;
    --sp-muted:      #A7A7A7;
    --sp-subtle:     #535353;
    --sp-border:     rgba(255,255,255,0.07);
}

html, body, [class*="css"] {
    font-family: 'Figtree', sans-serif !important;
    background-color: var(--sp-dark) !important;
    color: var(--sp-text) !important;
}

#MainMenu, footer, header { visibility: hidden; }

.block-container {
    padding: 0 !important;
    max-width: 100% !important;
    background: var(--sp-dark);
}

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background: var(--sp-black) !important;
    border-right: 1px solid var(--sp-border) !important;
    min-width: 260px !important;
    max-width: 260px !important;
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
}
section[data-testid="stSidebar"] > div {
    padding: 8px 0 !important;
    background: var(--sp-black) !important;
}
section[data-testid="stSidebar"] > div:first-child {
    padding: 8px 0 !important;
    background: var(--sp-black) !important;
}
section[data-testid="stSidebar"] .stSlider > div > div > div {
    background: var(--sp-green) !important;
}
section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] {
    background: var(--sp-subtle) !important;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {
    color: var(--sp-muted) !important;
    font-size: 12px !important;
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--sp-subtle); border-radius: 3px; }

/* ── SELECTBOX ── */
.stSelectbox > div > div {
    background: var(--sp-card-hover) !important;
    border: 1px solid var(--sp-subtle) !important;
    border-radius: 6px !important;
    color: var(--sp-text) !important;
    font-family: 'Figtree', sans-serif !important;
    font-size: 14px !important;
    font-weight: 500 !important;
}
.stSelectbox > div > div:hover {
    border-color: var(--sp-text) !important;
}

/* ── BUTTON ── */
.stButton > button {
    background: var(--sp-green) !important;
    color: #000 !important;
    border: none !important;
    border-radius: 500px !important;
    padding: 12px 32px !important;
    font-family: 'Figtree', sans-serif !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    letter-spacing: 0.02em !important;
    transition: transform 0.1s, background 0.15s !important;
    width: auto !important;
    min-width: 160px !important;
}
.stButton > button:hover {
    background: var(--sp-green-dark) !important;
    transform: scale(1.04) !important;
    color: #000 !important;
    border: none !important;
}

/* ── SPINNER ── */
.stSpinner > div { border-top-color: var(--sp-green) !important; }

/* ── ALERTS ── */
.stWarning { background: rgba(255,185,0,0.08) !important; border-left-color: #ffb900 !important; border-radius: 8px !important; }

/* ── PYPLOT ── */
.stPlotlyChart, .stPyplot { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── LOADERS ──────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        with open('music_recommendation_model.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        return None

@st.cache_resource
def load_feature_cols():
    try:
        with open('feature_cols.json', 'r') as f:
            return json.load(f)
    except:
        return None

@st.cache_data
def load_data():
    try:
        train   = pd.read_csv('train.csv', nrows=50000)
        songs   = pd.read_csv('songs.csv')
        members = pd.read_csv('members.csv')
        return train, songs, members
    except:
        return None, None, None

model        = load_model()
saved_fcols  = load_feature_cols()
train_df, songs_df, members_df = load_data()

icons = ["🎸","🎹","🎺","🥁","🎷","🎻","🪗","🎤","🎙️","🎚️",
         "🎛️","📻","🎼","🎵","🎶","🪘","🪕","🎧","🔊","🎟️"]

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo + branding
    st.markdown("""
    <div style="padding:28px 24px 16px">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px">
        <svg width="28" height="28" viewBox="0 0 24 24">
          <circle cx="12" cy="12" r="12" fill="#1DB954"/>
          <path d="M17.9 10.9C14.7 9 9.35 8.8 6.3 9.75c-.5.15-1-.15-1.15-.6-.15-.5.15-1 .6-1.15C9.65 6.8 15.6 7 19.25 9.15c.45.25.6.85.35 1.3-.25.35-.85.5-1.3.25M17.75 13.6c-.2.3-.6.45-.95.25-2.65-1.6-6.7-2.1-9.85-1.15-.35.1-.75-.1-.85-.45-.1-.35.1-.75.45-.85 3.6-1.1 8.05-.55 11.1 1.35.3.15.45.6.1.85M15.45 16.2c-.15.25-.5.35-.75.2C12.1 14.9 8.8 14.55 4.9 15.4c-.3.05-.6-.15-.65-.45-.05-.3.15-.6.45-.65 4.25-.95 7.9-.5 10.75 1.2.25.1.35.45.1.7z" fill="black"/>
        </svg>
        <span style="font-size:20px;font-weight:800;color:#fff;letter-spacing:-0.5px">Muse <span style="color:#1DB954">AI</span></span>
      </div>
      <div style="font-size:11px;color:#535353;font-weight:500;letter-spacing:0.04em;padding-left:38px">music recommendation engine</div>
    </div>

    <div style="height:1px;background:rgba(255,255,255,0.07);margin:0 24px 8px"></div>

    <!-- NAV LINKS -->
    <div style="padding:8px 12px">
      <div style="display:flex;align-items:center;gap:14px;padding:10px 12px;
           border-radius:6px;background:rgba(255,255,255,0.08);margin-bottom:2px">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="#fff"><path d="M12 3L2 12h3v9h6v-5h2v5h6v-9h3z"/></svg>
        <span style="font-size:13px;font-weight:700;color:#fff">Home</span>
      </div>
      <div style="display:flex;align-items:center;gap:14px;padding:10px 12px;border-radius:6px;margin-bottom:2px;cursor:pointer">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="#a7a7a7"><path d="M15.5 14h-.79l-.28-.27A6.47 6.47 0 0 0 16 9.5 6.5 6.5 0 1 0 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14z"/></svg>
        <span style="font-size:13px;font-weight:600;color:#a7a7a7">Search</span>
      </div>
      <div style="display:flex;align-items:center;gap:14px;padding:10px 12px;border-radius:6px;cursor:pointer">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="#a7a7a7"><path d="M3 18h18v-2H3v2zm0-5h18v-2H3v2zm0-7v2h18V6H3z"/></svg>
        <span style="font-size:13px;font-weight:600;color:#a7a7a7">Your Library</span>
      </div>
    </div>

    <div style="height:1px;background:rgba(255,255,255,0.07);margin:8px 24px 16px"></div>

    <!-- RECENTLY PLAYED -->
    <div style="padding:0 24px;margin-bottom:10px">
      <div style="font-size:11px;font-weight:700;color:#a7a7a7;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:12px">Recently Played</div>
    </div>
    <div style="padding:0 12px">
      <div style="display:flex;align-items:center;gap:12px;padding:8px 12px;border-radius:6px;cursor:pointer;margin-bottom:2px">
        <div style="width:40px;height:40px;border-radius:4px;background:linear-gradient(135deg,#FF6B6B,#c084fc);flex-shrink:0;display:flex;align-items:center;justify-content:center;font-size:16px">🎵</div>
        <div style="min-width:0">
          <div style="font-size:13px;font-weight:600;color:#fff;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">Daily Mix 1</div>
          <div style="font-size:11px;color:#a7a7a7">Playlist</div>
        </div>
      </div>
      <div style="display:flex;align-items:center;gap:12px;padding:8px 12px;border-radius:6px;cursor:pointer;margin-bottom:2px">
        <div style="width:40px;height:40px;border-radius:4px;background:linear-gradient(135deg,#38bdf8,#6366f1);flex-shrink:0;display:flex;align-items:center;justify-content:center;font-size:16px">🎵</div>
        <div style="min-width:0">
          <div style="font-size:13px;font-weight:600;color:#fff;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">Chill Vibes</div>
          <div style="font-size:11px;color:#a7a7a7">Playlist</div>
        </div>
      </div>
      <div style="display:flex;align-items:center;gap:12px;padding:8px 12px;border-radius:6px;cursor:pointer;margin-bottom:2px">
        <div style="width:40px;height:40px;border-radius:4px;background:linear-gradient(135deg,#f59e0b,#ef4444);flex-shrink:0;display:flex;align-items:center;justify-content:center;font-size:16px">🔥</div>
        <div style="min-width:0">
          <div style="font-size:13px;font-weight:600;color:#fff;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">Hype Mode</div>
          <div style="font-size:11px;color:#a7a7a7">Playlist</div>
        </div>
      </div>
      <div style="display:flex;align-items:center;gap:12px;padding:8px 12px;border-radius:6px;cursor:pointer;margin-bottom:2px">
        <div style="width:40px;height:40px;border-radius:4px;background:linear-gradient(135deg,#1DB954,#0a6e2a);flex-shrink:0;display:flex;align-items:center;justify-content:center;font-size:16px">🌟</div>
        <div style="min-width:0">
          <div style="font-size:13px;font-weight:600;color:#fff;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">Discover Weekly</div>
          <div style="font-size:11px;color:#a7a7a7">Updated Monday</div>
        </div>
      </div>
      <div style="display:flex;align-items:center;gap:12px;padding:8px 12px;border-radius:6px;cursor:pointer;margin-bottom:2px">
        <div style="width:40px;height:40px;border-radius:4px;background:linear-gradient(135deg,#a78bfa,#ec4899);flex-shrink:0;display:flex;align-items:center;justify-content:center;font-size:16px">💜</div>
        <div style="min-width:0">
          <div style="font-size:13px;font-weight:600;color:#fff;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">Liked Songs</div>
          <div style="font-size:11px;color:#a7a7a7">342 songs</div>
        </div>
      </div>
      <div style="display:flex;align-items:center;gap:12px;padding:8px 12px;border-radius:6px;cursor:pointer;margin-bottom:2px">
        <div style="width:40px;height:40px;border-radius:4px;background:linear-gradient(135deg,#06b6d4,#0ea5e9);flex-shrink:0;display:flex;align-items:center;justify-content:center;font-size:16px">🎧</div>
        <div style="min-width:0">
          <div style="font-size:13px;font-weight:600;color:#fff;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">Late Night Drive</div>
          <div style="font-size:11px;color:#a7a7a7">Playlist</div>
        </div>
      </div>
      <div style="display:flex;align-items:center;gap:12px;padding:8px 12px;border-radius:6px;cursor:pointer;margin-bottom:2px">
        <div style="width:40px;height:40px;border-radius:4px;background:linear-gradient(135deg,#d946ef,#f43f5e);flex-shrink:0;display:flex;align-items:center;justify-content:center;font-size:16px">🎤</div>
        <div style="min-width:0">
          <div style="font-size:13px;font-weight:600;color:#fff;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">Top Charts</div>
          <div style="font-size:11px;color:#a7a7a7">Global · Today</div>
        </div>
      </div>
    </div>

    <div style="height:1px;background:rgba(255,255,255,0.07);margin:16px 24px"></div>

    <!-- RECOMMENDATIONS SLIDER -->
    <div style="padding:0 24px;margin-bottom:6px">
      <div style="font-size:11px;font-weight:700;color:#a7a7a7;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:14px">Recommendations</div>
    </div>
    """, unsafe_allow_html=True)

    top_n = st.slider("Number of songs", 5, 20, 10, label_visibility="collapsed")
    st.markdown(f'<div style="padding:0 24px;font-size:12px;color:#1DB954;font-weight:700;margin-top:-8px;margin-bottom:4px">{top_n} songs selected</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:1px;background:rgba(255,255,255,0.07);margin:16px 24px"></div>', unsafe_allow_html=True)

    # Model info pills
    st.markdown('<div style="padding:0 24px"><div style="font-size:11px;font-weight:700;color:#a7a7a7;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:12px">Model Info</div></div>', unsafe_allow_html=True)
    for k, v, color in [
        ("Algorithm", "XGBoost",        "#1DB954"),
        ("Task",      "Classification", "#38bdf8"),
        ("Metric",    "AUC-ROC",        "#c084fc"),
        ("Target",    "1 = Replay",     "#f59e0b"),
    ]:
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;align-items:center;
             padding:10px 16px;margin:0 24px 8px;
             background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.07);
             border-radius:8px">
          <span style="font-size:12px;color:#a7a7a7">{k}</span>
          <span style="font-size:12px;font-weight:700;color:{color};font-family:monospace">{v}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div style="height:1px;background:rgba(255,255,255,0.07);margin:16px 24px"></div>', unsafe_allow_html=True)

    # Footer credit
    st.markdown("""
    <div style="padding:4px 24px 20px;font-size:11px;color:#535353;line-height:1.9">
      KKBOX Music Recommendation<br>Challenge — Kaggle<br>
      Built for <b style="color:#a7a7a7">Rhombix Technologies</b><br>
      ML Internship · Month 1
    </div>""", unsafe_allow_html=True)


# ─── MAIN CONTENT ─────────────────────────────────────────────────────────────
main = st.container()

with main:
    st.markdown('<div style="padding:24px 32px 0">', unsafe_allow_html=True)

    # ── TOP BAR ──
    st.markdown("""
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:32px">
      <div style="display:flex;gap:8px">
        <div style="width:32px;height:32px;border-radius:50%;background:rgba(0,0,0,0.6);
             display:flex;align-items:center;justify-content:center;cursor:pointer">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="#fff">
            <path d="M15.957 2.793a1 1 0 0 1 0 1.414L8.164 12l7.793 7.793a1 1 0 1 1-1.414 1.414L5.336 12l9.207-9.207a1 1 0 0 1 1.414 0z"/>
          </svg>
        </div>
        <div style="width:32px;height:32px;border-radius:50%;background:rgba(0,0,0,0.6);
             display:flex;align-items:center;justify-content:center;cursor:pointer">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="#fff">
            <path d="M8.043 2.793a1 1 0 0 1 1.414 0L18.664 12l-9.207 9.207a1 1 0 1 1-1.414-1.414L15.836 12 8.043 4.207a1 1 0 0 1 0-1.414z"/>
          </svg>
        </div>
      </div>
      <div style="width:32px;height:32px;border-radius:50%;background:#1DB954;
           display:flex;align-items:center;justify-content:center;
           font-size:13px;font-weight:800;color:#000;cursor:pointer">M</div>
    </div>
    """, unsafe_allow_html=True)

    # ── HERO ──
    st.markdown("""
    <div style="background:linear-gradient(180deg,#1a0a2e 0%,#121212 100%);
         border-radius:12px;padding:40px 40px 36px;margin-bottom:28px;position:relative;overflow:hidden">
      <div style="position:absolute;inset:0;background:radial-gradient(ellipse 60% 80% at 5% 50%,
           rgba(29,185,84,0.18) 0%,transparent 55%);pointer-events:none"></div>
      <div style="font-size:11px;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;
           color:#a7a7a7;margin-bottom:12px;display:flex;align-items:center;gap:8px">
        <span style="display:inline-block;width:20px;height:1px;background:#1DB954"></span>
        XGBoost · AI Recommendations · KKBOX
      </div>
      <div style="font-size:clamp(36px,5vw,58px);font-weight:900;line-height:1.05;
           letter-spacing:-2px;color:#fff;margin-bottom:12px">
        Your <span style="color:#1DB954">Daily</span><br>Mixes
      </div>
      <div style="font-size:13px;color:#a7a7a7;display:flex;align-items:center;gap:8px">
        <strong style="color:#fff">ML-powered</strong>
        <span style="opacity:0.4">•</span>
        <span>Personalised for you</span>
        <span style="opacity:0.4">•</span>
        <span>Updated today</span>
      </div>
      <div style="display:flex;gap:12px;margin-top:24px;align-items:center">
        <div style="background:#1DB954;color:#000;border-radius:500px;padding:13px 32px;
             font-size:14px;font-weight:700;cursor:pointer;display:inline-flex;align-items:center;gap:8px">
          ▶ Play All
        </div>
        <div style="background:transparent;color:#fff;border:1.5px solid #535353;border-radius:500px;
             padding:13px 28px;font-size:14px;font-weight:700;cursor:pointer;display:inline-flex;align-items:center;gap:8px">
          ⇄ Shuffle
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── DATA / MODEL LOGIC ─────────────────────────────────────────────────────
    if train_df is not None and model is not None:

        @st.cache_data
        def prepare_data(_train, _songs, _members):
            df = _train.merge(_songs, on='song_id', how='left')
            df = df.merge(_members, on='msno', how='left')
            num_cols = df.select_dtypes(include=[np.number]).columns
            df[num_cols] = df[num_cols].fillna(df[num_cols].median())
            cat_cols = df.select_dtypes(include=['object']).columns
            df[cat_cols] = df[cat_cols].fillna('Unknown')
            if 'bd' in df.columns:
                df['bd'] = df['bd'].apply(lambda x: x if 1 < x < 100 else np.nan)
                df['bd'].fillna(df['bd'].median(), inplace=True)
            if 'song_length' in df.columns:
                df['song_length_sec'] = df['song_length'] / 1000
            sc = df.groupby('song_id')['target'].count().reset_index()
            sc.columns = ['song_id', 'song_play_count']
            df = df.merge(sc, on='song_id', how='left')
            uc = df.groupby('msno')['target'].count().reset_index()
            uc.columns = ['msno', 'user_play_count']
            df = df.merge(uc, on='msno', how='left')
            le = LabelEncoder()
            for col in ['msno','song_id','source_system_tab','source_screen_name',
                        'source_type','genre_ids','artist_name','composer',
                        'lyricist','language','city','gender','registered_via']:
                if col in df.columns:
                    df[col] = le.fit_transform(df[col].astype(str))
            return df

        df = prepare_data(train_df, songs_df, members_df)

        if saved_fcols:
            feature_cols = [c for c in saved_fcols if c in df.columns]
        else:
            drop_cols = ['target','registration_init_time','expiration_date','song_length',
                         'song_replay_rate','user_replay_rate','artist_popularity',
                         'genre_replay_rate','source_replay_rate']
            feature_cols = [c for c in df.columns if c not in drop_cols]

        # ── STATS ROW ──
        replay_rate = df['target'].mean() * 100
        n_users     = df['msno'].nunique()
        n_songs     = df['song_id'].nunique()
        n_records   = len(df)

        st.markdown(f"""
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:28px">
          {"".join([
            f'''<div style="background:#181818;border-radius:10px;padding:20px 22px;
                 cursor:pointer;transition:background 0.15s;position:relative;overflow:hidden">
              <div style="position:absolute;top:0;left:0;right:0;height:2px;
                   background:linear-gradient(90deg,#1DB954,#38bdf8);opacity:0.6"></div>
              <div style="font-size:11px;font-weight:700;letter-spacing:0.1em;
                   text-transform:uppercase;color:#a7a7a7;margin-bottom:8px">{lbl}</div>
              <div style="font-size:26px;font-weight:900;color:#fff">{val}</div>
            </div>'''
            for lbl, val in [
                ("Records Loaded",  f"{n_records:,}"),
                ("Unique Users",    f"{n_users:,}"),
                ("Unique Songs",    f"{n_songs:,}"),
                ("Avg Replay Rate", f"{replay_rate:.1f}%"),
            ]
          ])}
        </div>
        """, unsafe_allow_html=True)

        # ── SELECT USER ──
        st.markdown("""
        <div style="font-size:11px;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;
             color:#a7a7a7;margin-bottom:16px;display:flex;align-items:center;gap:12px">
          Get Recommendations
          <div style="flex:1;height:1px;background:#282828"></div>
        </div>""", unsafe_allow_html=True)

        users = df['msno'].unique()[:200]
        col_a, col_b = st.columns([4, 1])
        with col_a:
            selected_user = st.selectbox("User ID", users, label_visibility="collapsed")
        with col_b:
            go = st.button("▶  Get My Recs")

        # ── RECOMMENDATIONS ──
        if go:
            with st.spinner("Analysing listening patterns..."):
                user_songs = df[df['msno'] == selected_user]['song_id'].unique()
                unseen     = [s for s in df['song_id'].unique() if s not in user_songs]

                if not unseen:
                    st.warning("This user has heard all songs in the dataset!")
                else:
                    sample    = df[df['song_id'].isin(unseen[:800])].drop_duplicates('song_id').copy()
                    user_data = df[df['msno'] == selected_user]
                    if not user_data.empty:
                        sample['msno']            = selected_user
                        sample['user_play_count'] = user_data['user_play_count'].values[0]

                    X_sample = sample[[c for c in feature_cols if c in sample.columns]].fillna(0)
                    for c in feature_cols:
                        if c not in X_sample.columns:
                            X_sample[c] = 0
                    X_sample = X_sample[feature_cols]

                    probs  = model.predict_proba(X_sample)[:, 1]
                    sample['replay_probability'] = probs
                    recs   = sample[['song_id','replay_probability','song_play_count']] \
                                   .sort_values('replay_probability', ascending=False) \
                                   .head(top_n)

            # ── TRACKLIST HEADER ──
            st.markdown(f"""
            <div style="font-size:11px;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;
                 color:#a7a7a7;margin:24px 0 16px;display:flex;align-items:center;gap:12px">
              Top {top_n} Recommendations · User {selected_user}
              <div style="flex:1;height:1px;background:#282828"></div>
            </div>
            <div style="display:grid;grid-template-columns:40px 1fr 100px 110px;
                 gap:16px;padding:0 16px 10px;border-bottom:1px solid #282828;margin-bottom:4px">
              <span style="font-size:12px;font-weight:700;color:#a7a7a7;text-transform:uppercase;letter-spacing:0.1em">#</span>
              <span style="font-size:12px;font-weight:700;color:#a7a7a7;text-transform:uppercase;letter-spacing:0.1em">Track</span>
              <span style="font-size:12px;font-weight:700;color:#a7a7a7;text-transform:uppercase;letter-spacing:0.1em;text-align:right">Plays</span>
              <span style="font-size:12px;font-weight:700;color:#a7a7a7;text-transform:uppercase;letter-spacing:0.1em;text-align:right">Confidence</span>
            </div>
            """, unsafe_allow_html=True)

            # ── TRACK ROWS ──
            for i, row in enumerate(recs.itertuples(), 1):
                pct   = row.replay_probability * 100
                plays = int(row.song_play_count) if not np.isnan(row.song_play_count) else 0
                plays_str = f"{plays/1000:.1f}K" if plays >= 1000 else str(plays)

                if pct >= 75:
                    badge_bg, badge_color = "rgba(29,185,84,0.15)", "#1DB954"
                elif pct >= 60:
                    badge_bg, badge_color = "rgba(56,189,248,0.15)", "#38bdf8"
                else:
                    badge_bg, badge_color = "rgba(255,255,255,0.07)", "#a7a7a7"

                bar_w = min(pct, 100)
                icon  = icons[(i - 1) % len(icons)]

                st.markdown(f"""
                <div style="display:grid;grid-template-columns:40px 1fr 100px 110px;gap:16px;
                     padding:10px 16px;align-items:center;border-radius:6px;cursor:pointer;
                     transition:background 0.12s;"
                     onmouseover="this.style.background='#282828'"
                     onmouseout="this.style.background='transparent'">

                  <div style="font-size:14px;color:#a7a7a7;text-align:center;font-weight:500">{i}</div>

                  <div style="display:flex;align-items:center;gap:12px;min-width:0">
                    <div style="width:40px;height:40px;border-radius:4px;background:#282828;
                         display:flex;align-items:center;justify-content:center;
                         font-size:16px;flex-shrink:0">{icon}</div>
                    <div style="min-width:0">
                      <div style="font-size:14px;font-weight:600;color:#fff;white-space:nowrap;
                           overflow:hidden;text-overflow:ellipsis">Song ID · {row.song_id}</div>
                      <div style="font-size:12px;color:#a7a7a7;margin-top:2px">KKBOX · XGBoost Rec</div>
                    </div>
                  </div>

                  <div style="font-size:13px;color:#a7a7a7;text-align:right">{plays_str}</div>

                  <div style="display:flex;align-items:center;gap:8px;justify-content:flex-end">
                    <div style="width:52px;height:3px;background:#535353;border-radius:3px;overflow:hidden">
                      <div style="width:{bar_w}%;height:100%;background:#1DB954;border-radius:3px"></div>
                    </div>
                    <span style="background:{badge_bg};color:{badge_color};
                         padding:3px 10px;border-radius:500px;font-size:12px;
                         font-weight:700;white-space:nowrap">{pct:.1f}%</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)

            # ── CHART ──
            st.markdown("""
            <div style="font-size:11px;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;
                 color:#a7a7a7;margin:28px 0 16px;display:flex;align-items:center;gap:12px">
              Probability Distribution
              <div style="flex:1;height:1px;background:#282828"></div>
            </div>""", unsafe_allow_html=True)

            fig, ax = plt.subplots(figsize=(11, max(4, top_n * 0.58)))
            fig.patch.set_facecolor('#181818')
            ax.set_facecolor('#181818')

            labels     = [f"Song {r.song_id}" for r in recs.itertuples()]
            probs_list = [r.replay_probability * 100 for r in recs.itertuples()]
            colors     = ['#1DB954' if p >= 75 else '#38bdf8' if p >= 60 else '#535353'
                          for p in probs_list]

            bars = ax.barh(labels[::-1], probs_list[::-1],
                           color=colors[::-1], height=0.55, edgecolor='none')
            for bar, prob in zip(bars, probs_list[::-1]):
                ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
                        f'{prob:.1f}%', va='center', color='#a7a7a7',
                        fontsize=9, fontfamily='monospace')

            ax.set_xlabel('Replay Probability (%)', color='#535353',
                          fontsize=9, labelpad=10)
            ax.tick_params(colors='#535353', labelsize=9)
            ax.set_xlim(0, 115)
            for sp in ['top', 'right', 'bottom']:
                ax.spines[sp].set_visible(False)
            ax.spines['left'].set_color('#282828')
            ax.xaxis.grid(True, color='#282828', linewidth=0.5, alpha=0.8)
            ax.set_axisbelow(True)
            ax.set_title(f'Predicted Replay Probability — User {selected_user}',
                         color='#fff', fontsize=11, pad=16, loc='left',
                         fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    else:
        # ─── DEMO MODE (no files) ──────────────────────────────────────────────
        st.markdown("""
        <div style="background:#282828;border:1px solid rgba(255,185,0,0.2);border-radius:10px;
             padding:14px 18px;margin-bottom:24px;font-size:13px;color:#ffb900;font-weight:500">
          ⚠️  Demo Mode — place <code style="background:rgba(255,255,255,0.1);padding:2px 6px;
          border-radius:4px">train.csv</code>, <code style="background:rgba(255,255,255,0.1);
          padding:2px 6px;border-radius:4px">songs.csv</code>, <code style="background:rgba(255,255,255,0.1);
          padding:2px 6px;border-radius:4px">members.csv</code> and <code style="background:rgba(255,255,255,0.1);
          padding:2px 6px;border-radius:4px">music_recommendation_model.pkl</code> in this folder, then rerun.
        </div>
        """, unsafe_allow_html=True)

        np.random.seed(42)
        demo_songs = [f"Song {100000 + i * 7919}" for i in range(top_n)]
        demo_probs = sorted(np.random.uniform(0.52, 0.91, top_n), reverse=True)
        demo_plays = np.random.randint(1000, 80000, top_n)

        # Demo stats
        st.markdown("""
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:28px">
          <div style="background:#181818;border-radius:10px;padding:20px 22px;position:relative;overflow:hidden">
            <div style="position:absolute;top:0;left:0;right:0;height:2px;
                 background:linear-gradient(90deg,#1DB954,#38bdf8);opacity:0.6"></div>
            <div style="font-size:11px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#a7a7a7;margin-bottom:8px">Songs in Library</div>
            <div style="font-size:26px;font-weight:900;color:#fff">7.4M</div>
          </div>
          <div style="background:#181818;border-radius:10px;padding:20px 22px;position:relative;overflow:hidden">
            <div style="position:absolute;top:0;left:0;right:0;height:2px;
                 background:linear-gradient(90deg,#1DB954,#38bdf8);opacity:0.6"></div>
            <div style="font-size:11px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#a7a7a7;margin-bottom:8px">Active Users</div>
            <div style="font-size:26px;font-weight:900;color:#fff">34K</div>
          </div>
          <div style="background:#181818;border-radius:10px;padding:20px 22px;position:relative;overflow:hidden">
            <div style="position:absolute;top:0;left:0;right:0;height:2px;
                 background:linear-gradient(90deg,#1DB954,#38bdf8);opacity:0.6"></div>
            <div style="font-size:11px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#a7a7a7;margin-bottom:8px">Unique Songs</div>
            <div style="font-size:26px;font-weight:900;color:#fff">2.3M</div>
          </div>
          <div style="background:#181818;border-radius:10px;padding:20px 22px;position:relative;overflow:hidden">
            <div style="position:absolute;top:0;left:0;right:0;height:2px;
                 background:linear-gradient(90deg,#1DB954,#38bdf8);opacity:0.6"></div>
            <div style="font-size:11px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#a7a7a7;margin-bottom:8px">Avg Replay Rate</div>
            <div style="font-size:26px;font-weight:900;color:#fff">68%</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="font-size:11px;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;
             color:#a7a7a7;margin-bottom:16px;display:flex;align-items:center;gap:12px">
          Demo Recommendations
          <div style="flex:1;height:1px;background:#282828"></div>
        </div>
        <div style="display:grid;grid-template-columns:40px 1fr 100px 110px;
             gap:16px;padding:0 16px 10px;border-bottom:1px solid #282828;margin-bottom:4px">
          <span style="font-size:12px;font-weight:700;color:#a7a7a7;text-transform:uppercase;letter-spacing:0.1em">#</span>
          <span style="font-size:12px;font-weight:700;color:#a7a7a7;text-transform:uppercase;letter-spacing:0.1em">Track</span>
          <span style="font-size:12px;font-weight:700;color:#a7a7a7;text-transform:uppercase;letter-spacing:0.1em;text-align:right">Plays</span>
          <span style="font-size:12px;font-weight:700;color:#a7a7a7;text-transform:uppercase;letter-spacing:0.1em;text-align:right">Confidence</span>
        </div>
        """, unsafe_allow_html=True)

        for i, (song, prob, plays) in enumerate(zip(demo_songs, demo_probs, demo_plays), 1):
            pct = prob * 100
            plays_str = f"{plays/1000:.1f}K" if plays >= 1000 else str(plays)

            if pct >= 75:
                badge_bg, badge_color = "rgba(29,185,84,0.15)", "#1DB954"
            elif pct >= 60:
                badge_bg, badge_color = "rgba(56,189,248,0.15)", "#38bdf8"
            else:
                badge_bg, badge_color = "rgba(255,255,255,0.07)", "#a7a7a7"

            bar_w = min(pct, 100)
            icon  = icons[(i - 1) % len(icons)]

            st.markdown(f"""
            <div style="display:grid;grid-template-columns:40px 1fr 100px 110px;gap:16px;
                 padding:10px 16px;align-items:center;border-radius:6px;cursor:pointer">
              <div style="font-size:14px;color:#a7a7a7;text-align:center;font-weight:500">{i}</div>
              <div style="display:flex;align-items:center;gap:12px;min-width:0">
                <div style="width:40px;height:40px;border-radius:4px;background:#282828;
                     display:flex;align-items:center;justify-content:center;
                     font-size:16px;flex-shrink:0">{icon}</div>
                <div style="min-width:0">
                  <div style="font-size:14px;font-weight:600;color:#fff;white-space:nowrap;
                       overflow:hidden;text-overflow:ellipsis">{song}</div>
                  <div style="font-size:12px;color:#a7a7a7;margin-top:2px">KKBOX · XGBoost Rec</div>
                </div>
              </div>
              <div style="font-size:13px;color:#a7a7a7;text-align:right">{plays_str}</div>
              <div style="display:flex;align-items:center;gap:8px;justify-content:flex-end">
                <div style="width:52px;height:3px;background:#535353;border-radius:3px;overflow:hidden">
                  <div style="width:{bar_w}%;height:100%;background:#1DB954;border-radius:3px"></div>
                </div>
                <span style="background:{badge_bg};color:{badge_color};
                     padding:3px 10px;border-radius:500px;font-size:12px;
                     font-weight:700;white-space:nowrap">{pct:.1f}%</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ── FOOTER ──
    st.markdown("""
    <div style="text-align:center;padding:40px 0 20px;font-size:11px;color:#535353;
         letter-spacing:0.08em;border-top:1px solid rgba(255,255,255,0.07);margin-top:3rem">
      Muse AI &nbsp;·&nbsp; <span style="color:#1DB954">Rhombix Technologies</span>
      &nbsp;ML Internship &nbsp;·&nbsp; XGBoost + Streamlit &nbsp;·&nbsp; KKBOX Dataset
    </div>
    </div>
    """, unsafe_allow_html=True)