import time
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import base64
import os

# =========================
# App Config
# =========================
st.set_page_config(
    page_title="T20 Final Score Predictor",
    page_icon="🏏",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# =========================
# Background Image Loader
# =========================
def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded_string}");
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local("new_bgg.jpg")

team_code_map = {
    "Afghanistan": "afg",
    "Australia": "aus",
    "Bangladesh": "ban",
    "Canada": "can",
    "England": "eng",
    "Hong Kong": "hk",
    "India": "bcci",
    "Ireland": "ire",
    "Kenya": "ken",
    "Namibia": "nam",
    "Nepal": "nep",
    "Netherlands": "net",
    "New Zealand": "nz",
    "Pakistan": "pak",
    "Papua New Guinea": "png",
    "South Africa": "sa",
    "Scotland": "sco",
    "Sri Lanka": "sl",
    "United Arab Emirates": "uae",
    "West Indies": "wi",
    "Zimbabwe": "zim"
}

def show_team_logo(team_name):
    team_code = team_code_map.get(team_name, "").lower()
    logo_path = f"Imagess/{team_code}.png"
    if os.path.exists(logo_path):
        st.image(logo_path, width=100)
    else:
        st.write("🏳️")


# =========================
# Global Styles
# =========================

st.markdown(
    """
    <style>
      :root {
        --glass-bg: rgba(255, 255, 255, 0.08);
        --glass-border: rgba(255, 255, 255, 0.15);
        --text: #f5f5f5;
        --muted: #cbd5e1;
        --primary: #00f5d4;
        --secondary: #00bbf9;
      }

      /* Glassmorphism card */
      .glass-card {
        background: var(--glass-bg);
        border: 1px solid var(--glass-border);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.25);
        backdrop-filter: blur(8px);
      }

      /* Neon gradient heading */
      .neon-text {
        font-size: 34px;
        font-weight: 900;
        background: linear-gradient(90deg, var(--primary), var(--secondary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0px 0px 8px rgba(0,245,212,0.6);
      }

      /* Animated buttons */
      div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        color: black;
        font-weight: 700;
        border-radius: 12px;
        padding: 0.6em 1em;
        transition: all 0.2s ease;
        box-shadow: 0 0 12px rgba(0,245,212,0.4);
      }
      div.stButton > button[kind="primary"]:hover {
        transform: scale(1.05);
        box-shadow: 0 0 20px rgba(0,245,212,0.6);
      }

      /* Chips */
      .chip {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        background: rgba(255,255,255,0.12);
        font-size: 13px;
        margin: 4px;
      }

      /* Result pulse */
      .pulse-wrap { position: relative; display: inline-block; }
      .pulse-ring {
        position: absolute; inset: -14px;
        border-radius: 9999px;
        border: 2px solid rgba(0,245,212,.4);
        animation: pulse 2.2s infinite;
      }
      @keyframes pulse {
        0% { transform: scale(0.95); opacity: .45; }
        70% { transform: scale(1); opacity: .06; }
        100% { transform: scale(1.05); opacity: 0; }
      }

      h1, h2, h3, label, p, span, div {
        color: var(--text) !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# Load Model and Data
# =========================
@st.cache_resource(show_spinner=False)
def load_assets():
    model_ = joblib.load("final_rf_model.pkl")
    encoders_ = joblib.load("label_encoders.pkl")
    player_stats_ = pd.read_csv("player_stats.csv")
    df_cleaned_ = pd.read_csv("df_cleaned.csv")
    return model_, encoders_, player_stats_, df_cleaned_

with st.spinner("Booting cricket brain 🧠🏏..."):
    model, label_encoders, player_stats, df_cleaned = load_assets()

# =========================
# Header
# =========================
st.markdown(
    """
    <div style="text-align:center;">
      <div class="chip">⚡ Live T20 Predictor • 5+ overs</div>
      <h1 class="neon-text">T20 Final Score Predictor</h1>
      <p style="color: var(--muted);">Predict the final score with live match inputs. Nail the chase! 🚀</p>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================
# Match Context
# =========================
with st.container():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🧠 Match Context")
    col1, col2 = st.columns(2)

    all_teams = sorted(label_encoders["batting_team"].classes_)

    with col1:
        batting_team = st.selectbox(
            "Batting Team 🏏",
            all_teams,
            key="batting_team_sel"
        )
        show_team_logo(batting_team)
    # Dynamically filter the bowling team list to exclude the selected batting team
    available_bowling_teams = [team for team in all_teams if team != batting_team]

    with col2:
        bowling_team = st.selectbox(
            "Bowling Team 🎯",
            available_bowling_teams,
            key="bowling_team_sel"
        )
        show_team_logo(bowling_team)
    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Filter valid players for the selected teams
# =========================
batters_for_team = sorted(
    player_stats[player_stats["player_name"].isin(
        df_cleaned[df_cleaned["batting_team"] == batting_team]["batter"].unique()
    )]["player_name"].unique()
)
bowlers_for_team = sorted(
    player_stats[player_stats["player_name"].isin(
        df_cleaned[df_cleaned["bowling_team"] == bowling_team]["bowler"].unique()
    )]["player_name"].unique()
)

# =========================
# Player Inputs
# =========================
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("👥 Player Inputs")

pi1, pi2 = st.columns(2)

with pi1:
    batter = st.selectbox(
        "Striker (Batter) ⚔️",
        batters_for_team,
        key="batter_sel"
    )

# Dynamically filter the non-striker list to exclude the selected striker
available_non_strikers = [p for p in batters_for_team if p != batter]

with pi2:
    non_striker = st.selectbox(
        "Non-Striker 🤝",
        available_non_strikers,  # Use the filtered list
        key="non_striker_sel"
    )

# The swap button is no longer needed as the selections are tied to the dropdowns directly
# and the session state management for swapping becomes much more complex with filtering.
# For a better UX, direct selection is more intuitive than swap->rerun.

bowler = st.selectbox("Current Bowler 🧤", bowlers_for_team, key="bowler_sel")
st.markdown('</div>', unsafe_allow_html=True)
# =========================
# Match State Inputs
# =========================
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("🎯 Match State Inputs")
ms1, ms2, ms3 = st.columns(3)
with ms1:
    over_number = st.number_input("Over Number ⏱️", min_value=5, max_value=19, value=6)
with ms2:
    ball_number = st.number_input("Ball in Over ⚪", min_value=1, max_value=6, value=2)
with ms3:
    current_score = st.number_input("Current Score 🧮", min_value=0, value=50)
ms4, ms5 = st.columns(2)
with ms4:
    last_five = st.number_input("Runs in Last 30 Balls 🔥", min_value=0, value=40)
with ms5:
    wickets = st.number_input("Wickets Lost ❌", min_value=0, max_value=9, value=1)

st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Helpers
# =========================
def get_player_features(name: str, role: str):
    row = player_stats[player_stats["player_name"] == name]
    if row.empty:
        return (25.0, 120.0) if role == "batter" else (30.0, 8.0)
    return (float(row["batting_avg"]), float(row["strike_rate"])) if role == "batter" else (float(row["bowling_avg"]), float(row["economy"]))

# =========================
# Predict Action
# =========================
predict_clicked = st.button("🎯 Predict Final Score", use_container_width=True, type="primary")

if predict_clicked:
    batting_encoded = label_encoders["batting_team"].transform([batting_team])[0]
    bowling_encoded = label_encoders["bowling_team"].transform([bowling_team])[0]

    batter_avg, batter_sr = get_player_features(batter, "batter")
    non_striker_avg, non_striker_sr = get_player_features(non_striker, "batter")
    bowler_avg, bowler_eco = get_player_features(bowler, "bowler")

    balls_bowled = int(over_number * 6 + ball_number)
    balls_left = max(0, 120 - balls_bowled)
    crr = current_score / (over_number + (ball_number - 1) / 6)
    wickets_left = 10 - wickets

    input_data = pd.DataFrame([{
        "batting_team": batting_encoded,
        "bowling_team": bowling_encoded,
        "over_number": over_number,
        "ball_number": ball_number,
        "current_score": current_score,
        "wickets": wickets,
        "balls_left": balls_left,
        "crr": round(crr, 2),
        "last_five": last_five,
        "wickets_left": wickets_left,
        "batter_avg": batter_avg,
        "batter_sr": batter_sr,
        "non_striker_avg": non_striker_avg,
        "non_striker_sr": non_striker_sr,
        "bowler_avg": bowler_avg,
        "bowler_eco": bowler_eco,
    }])

    with st.spinner("Crunching the numbers… 📊"):
        time.sleep(1)
    predicted_score = int(model.predict(input_data)[0])
    band = max(8, int(0.07 * max(100, predicted_score)))
    lo, hi = max(current_score, predicted_score - band), predicted_score + band

    st.markdown(
        f"""
        <div class="glass-card" style="text-align:center;">
            <div style="font-size:14px; color: var(--muted); text-transform: uppercase;">Predicted Final Score</div>
            <div class="pulse-wrap" style="margin-top:6px;">
                <div class="pulse-ring"></div>
                <div style="font-size:56px; font-weight:900; background:linear-gradient(90deg,#00f5d4,#00bbf9); -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                    {predicted_score} <span style="font-size:22px; color:var(--muted) !important;">runs</span>
                </div>
            </div>
            <div style="margin-top:14px;">
                <span class="chip">🔎 Range <b>{lo} - {hi}</b></span>
                <span class="chip">📈 Projected RR <b>{predicted_score/20:.2f}</b></span>
                <span class="chip">🧱 Wkts Left <b>{wickets_left}</b></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


