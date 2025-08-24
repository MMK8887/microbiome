from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Optional deep NLP (preferred) ---
USE_ST_EMBEDDINGS = True
ST_MODEL_NAME = "all-MiniLM-L6-v2"

# -------------------------------
# 1) Constants & UI setup
# -------------------------------
SCORE_MIN, SCORE_MAX = 0.0, 10.0
THRESH_GREEN = 7.0
THRESH_YELLOW = 5.0

st.set_page_config(page_title="MIOME — Gut Health Assistant", page_icon="🤖", layout="centered")

# -------------------------------
# 2) Helpers (scores, feedback)
# -------------------------------
def feedback(score: float, good: str, mid: str, low: str) -> str:
    if score >= THRESH_GREEN: 
        return good
    if score >= THRESH_YELLOW: 
        return mid
    return low

def clip_scores(df: pd.DataFrame, score_cols: list) -> pd.DataFrame:
    for c in score_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").clip(SCORE_MIN, SCORE_MAX)
    return df.dropna(subset=score_cols)

def compute_gut_score_from_values(fiber: float, probiotic: float, diversity: float) -> float:
    return round(0.4 * fiber + 0.3 * probiotic + 0.3 * diversity, 2)

def gut_zone(score: float) -> str:
    if score >= THRESH_GREEN: 
        return "Green"
    if score >= THRESH_YELLOW: 
        return "Yellow"
    return "Red"

def rule_based_reco(scores: list) -> str:
    fiber, probiotic, diversity = scores
    recs = []
    if fiber < 5: 
        recs.append("🌾 Increase fiber with oats, beans, lentils, whole grains, and veggies.")
    if probiotic < 5: 
        recs.append("🦠 Add probiotics: yogurt, kefir, kimchi, sauerkraut, kombucha.")
    if diversity < 5: 
        recs.append("🥗 Increase plant diversity — aim for 20–30 different plants per week.")
    return " ".join(recs) if recs else "✅ All scores look good! Maintain your current routine."

# -------------------------------
# 3) Session storage
# -------------------------------
if "user_df" not in st.session_state: 
    st.session_state.user_df = pd.DataFrame()
if "scores" not in st.session_state: 
    st.session_state.scores = None
if "history" not in st.session_state: 
    st.session_state.history = []
if "user_name" not in st.session_state: 
    st.session_state.user_name = "Guest"

# -------------------------------
# 4) Intent inventory & Model
# -------------------------------
INTENT_CANON = {
    "greetings": ["hello", "hi", "hey", "hola", "yo", "good morning", "good evening",
                  "hey there", "how are you", "how are you doing", "what's up"],
    "report_summary": ["report", "summary", "summarise", "summarize",
                       "how am i doing", "what are my scores", "overall",
                       "show results", "latest summary", "how's my health", "status"],
    "diet": ["diet", "what do i eat", "food suggestions", "meal ideas",
             "how's my diet", "nutrition", "food plan"],
    "improvements": ["improve", "improvements", "tips", "how to get better",
                     "optimize", "recommendations", "next steps", "what should i change",
                     "what can i do to improve"],
    "fiber_info": ["fiber", "fiber intake", "fiber foods", "is my fiber okay", "increase fiber",
                   "tell me about fiber", "what foods have fiber"],
    "probiotic_info": ["probiotic", "probiotics", "yogurt", "fermented foods", "add probiotics",
                       "what is kefir", "what is kimchi", "what is miso", "what is sauerkraut",
                       "what is tempeh", "tell me about fermented foods", "ferment", "fermented"],
    "diversity_info": ["diversity", "microbiome diversity", "variety", "food variety", "plant diversity"],
    "rainbow_plate_info": ["what is a rainbow plate", "rainbow plate", "what is rainbow plate",
                           "explain rainbow plate", "tell me about rainbow plate"],
    "help": ["help", "what can you do", "how to use", "features", "guide", "what do you do"],
    "thanks": ["thank you", "thanks", "thanks so much", "appreciate it", "got it", "thankyou"],
    "goodbye": ["bye", "goodbye", "see you later", "catch you later"],
    "follow_up": ["and also", "anything else", "what else", "also", "what about",
                  "next", "another tip", "what else can i do", "yes", "yeah", "yup", "sure"]
}

INTENT_PHRASES = [p for phrases in INTENT_CANON.values() for p in phrases]
INTENT_OF_PHRASE = [label for label, phrases in INTENT_CANON.items() for _ in phrases]

@st.cache_resource
def get_intent_model():
    try:
        # Try to use sentence transformers if available
        import torch
        from sentence_transformers import SentenceTransformer, util as st_util
        embedder = SentenceTransformer(ST_MODEL_NAME)
        intent_emb = embedder.encode(INTENT_PHRASES, convert_to_tensor=True)
        return embedder, st_util, intent_emb, None
    except ImportError:
        # Fallback to TF-IDF
        vec = TfidfVectorizer().fit(INTENT_PHRASES)
        intent_emb = vec.transform(INTENT_PHRASES)
        return None, None, intent_emb, vec

embedder, util, INTENT_EMB, VEC = get_intent_model()

def predict_intent(text: str) -> str:
    text = (text or "").strip().lower()
    if not text: 
        return "unknown"
    
    if embedder is not None:
        q = embedder.encode([text], convert_to_tensor=True)
        sims = util.cos_sim(q, INTENT_EMB).flatten()
        best = int(sims.argmax().item())
        return INTENT_OF_PHRASE[best] if float(sims[best]) >= 0.45 else "unknown"
    else:
        q = VEC.transform([text])
        sims = cosine_similarity(q, INTENT_EMB).ravel()
        best = int(np.argmax(sims))
        return INTENT_OF_PHRASE[best] if float(sims[best]) >= 0.35 else "unknown"

# --- Knowledge base ---
PROBIOTIC_FOOD_INFO = {
    "kefir": "🥛 Kefir is a fermented milk drink, packed with diverse probiotics.",
    "kimchi": "🌶️ Kimchi is a Korean side dish made from fermented vegetables, rich in probiotics.",
    "miso": "🍲 Miso is a Japanese seasoning from fermented soybeans, gut-friendly.",
    "sauerkraut": "🥬 Sauerkraut is fermented cabbage, full of probiotics.",
    "tempeh": "🍱 Tempeh is an Indonesian fermented soybean product, rich in protein & probiotics.",
    "fermented food": "Fermented foods like kimchi, kefir, sauerkraut, miso, and tempeh support gut health."
}

# -------------------------------
# 5) Respond function
# -------------------------------
def respond(intent: str, text: str) -> str:
    scores = st.session_state.scores
    user = st.session_state.user_name
    t = (text or "").strip().lower()

    # Handle follow-up questions (yes/yeah responses)
    if intent == "follow_up" and t in ["yes", "yeah", "yup", "sure"] and scores is not None:
        intent = "report_summary"

    # If no data uploaded
    if scores is None and intent not in ["greetings", "thanks", "goodbye", "help"]:
        return "📂 Please upload your data first. Then ask about summary, diet, improvements, or fiber/probiotics."

    # Special cases
    if intent == "greetings":
        if scores is None:
            return "👋 Hello there! Upload a file and I'll generate your gut health report."
        else:
            return f"👋 Hello, {user}! Your last Gut Score was {scores['gut_score']:.1f}/10. Want a full summary?"
    
    if intent == "thanks":
        return "🙏 You're welcome! Feel free to ask more questions whenever you need."
    
    if intent == "goodbye":
        return "👋 Goodbye! Take care of your gut health."

    # Probiotic food direct lookup
    for food_name, info in PROBIOTIC_FOOD_INFO.items():
        if food_name in t or (food_name.replace(" ", "") in t.replace(" ", "")) or (food_name[:-1] in t):
            return info
    
    # If we have scores, proceed with analysis
    if scores is None:
        return "📂 Please upload your data first to get personalized recommendations."
    
    f, p, d = scores["fiber"], scores["probiotic"], scores["diversity"]
    g = scores["gut_score"]
    zone = gut_zone(g)
    reco = rule_based_reco([f, p, d])

    if intent == "report_summary":
        return (
            f"📊 Report Summary for {user}:\n"
            f"- 🌾 Fiber: {f:.1f}/10\n"
            f"- 🦠 Probiotic: {p:.1f}/10\n"
            f"- 🥗 Diversity: {d:.1f}/10\n"
            f"➡️ Gut Score: {g:.1f}/10 ({zone} zone)\n\n"
            f"💡 Tip: {reco}"
        )
    
    if intent == "diet":
        return f"🥗 Diet snapshot — Fiber {f:.1f}, Probiotic {p:.1f}, Diversity {d:.1f}. {reco}"
    
    if intent == "improvements":
        parts = []
        if f < 7: 
            parts.append("🌾 Fiber: add legumes, whole grains, veggies, and fruit skins.")
        if p < 7: 
            parts.append("🦠 Probiotics: include yogurt/kefir or fermented foods daily.")
        if d < 7: 
            parts.append("🥗 Diversity: rotate 20–30 plant foods weekly (herbs, nuts, seeds).")
        
        if not parts: 
            parts.append("✅ You're in a good range—keep your variety and hydration steady.")
        
        return "Improvements:\n- " + "\n- ".join(parts)
    
    if intent == "fiber_info":
        return f"🌾 Fiber score {f:.1f}/10. Aim for ~25–35g/day. Add oats, beans, lentils, chia/flax, veggies."
    
    if intent == "probiotic_info":
        return f"🦠 Probiotic score {p:.1f}/10. Include yogurt/kefir daily; add kimchi/sauerkraut/miso/tempeh weekly."
    
    if intent == "diversity_info":
        return f"🥗 Diversity score {d:.1f}/10. Try a 'rainbow plate' and swap staples weekly."
    
    if intent == "rainbow_plate_info":
        return "🌈 A 'rainbow plate' means eating a wide variety of plant foods to get different nutrients. Fill your plate with many colors."
    
    if intent == "help":
        return ("💡 I analyze your uploaded file and answer about: summary, diet, fiber, probiotics, diversity, and improvements.\n"
                "Try asking: 'How's my diet?' or 'Show me improvements'.")
    
    return "🤔 I can help with summary, diet, improvements, fiber, probiotics, or diversity."

# -------------------------------
# 6) UI — Upload
# -------------------------------
st.title("🤖 MIOME")
st.caption("Your Gut Health Assistant")

with st.sidebar:
    st.header("📥 Upload Data")
    user = st.text_input("👤 Your name (optional)", value=st.session_state.user_name).strip() or "Guest"
    st.session_state.user_name = user
    uploaded = st.file_uploader("📄 Upload CSV or XLSX", type=["csv", "xlsx"])

if uploaded:
    try:
        # Read file
        if uploaded.name.endswith(".xlsx"):
            raw = pd.read_excel(uploaded)
        else:
            raw = pd.read_csv(uploaded)
        
        # Check for required columns (flexible naming)
        required = ["fiber_score", "probiotic_score", "diversity_score"]
        if not all(c in raw.columns for c in required):
            # Try alternative naming
            required_alt = ["Fiber", "Probiotic", "Diversity"]
            if all(c in raw.columns for c in required_alt):
                raw = raw.rename(columns={
                    "Fiber": "fiber_score", 
                    "Probiotic": "probiotic_score", 
                    "Diversity": "diversity_score"
                })
            else:
                st.error("❌ Missing required columns. Expected: fiber_score, probiotic_score, diversity_score (or Fiber, Probiotic, Diversity)")
                st.stop()
        
        # Handle dates if present
        if "date" in raw.columns:
            raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
            raw = raw.dropna(subset=["date"]).sort_values("date")
        
        # Clean and validate scores
        raw = clip_scores(raw, ["fiber_score", "probiotic_score", "diversity_score"])
        
        if raw.empty:
            st.error("❌ No valid data found after processing")
            st.stop()
        
        st.session_state.user_df = raw.copy()
        
        # Calculate scores from latest entry
        latest = raw.iloc[-1]
        f = float(latest["fiber_score"])
        p = float(latest["probiotic_score"])
        d = float(latest["diversity_score"])
        g = compute_gut_score_from_values(f, p, d)
        
        st.session_state.scores = {
            "fiber": f, 
            "probiotic": p, 
            "diversity": d, 
            "gut_score": g
        }
        
        st.success("✅ File uploaded and analyzed successfully!")
        
    except Exception as e:
        st.error(f"❌ Failed to read file: {str(e)}")
        st.stop()

# -------------------------------
# 7) Report Display
# -------------------------------
scores = st.session_state.scores
if scores is not None:
    zone = gut_zone(scores["gut_score"])
    zone_color = {"Green": "🟢", "Yellow": "🟡", "Red": "🔴"}
    
    st.subheader(f"Hello, {st.session_state.user_name}!")
    st.info(f"{zone_color.get(zone, '⚪')} Overall Gut Score: {scores['gut_score']:.1f}/10 — {zone} zone")
    
    # Display metrics in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fiber_feedback = feedback(scores['fiber'], "Good", "Needs attention", "Needs urgent help")
        st.metric("🌾 Fiber Score", f"{scores['fiber']:.1f}/10", delta=fiber_feedback)
    
    with col2:
        probiotic_feedback = feedback(scores['probiotic'], "Good", "Needs attention", "Needs urgent help")
        st.metric("🦠 Probiotic Score", f"{scores['probiotic']:.1f}/10", delta=probiotic_feedback)
    
    with col3:
        diversity_feedback = feedback(scores['diversity'], "Good", "Needs attention", "Needs urgent help")
        st.metric("🥗 Diversity Score", f"{scores['diversity']:.1f}/10", delta=diversity_feedback)
    
    # Show personalized recommendation
    st.markdown("**🎯 Personalized Recommendation:**")
    recommendation = rule_based_reco([scores['fiber'], scores['probiotic'], scores['diversity']])
    st.success(recommendation)
    
else:
    st.info("📂 Upload a CSV or XLSX file to see your gut health report and start chatting!")

st.markdown("---")

# -------------------------------
# 8) Chatbot Interface
# -------------------------------
st.subheader("💬 Chat with MIOME")

# Display chat history (limit to last 12 messages for performance)
for who, msg in st.session_state.history[-12:]:
    with st.chat_message("user" if who == "You" else "assistant"):
        st.markdown(msg)

# Chat input
user_msg = st.chat_input("Ask about summary, diet, fiber, probiotics, diversity, or improvements…")

if user_msg:
    # Predict intent and generate response
    intent = predict_intent(user_msg)
    answer = respond(intent, user_msg)
    
    # Add to history
    st.session_state.history.append(("You", user_msg))
    st.session_state.history.append(("Bot", answer))
    
    # Rerun to show new messages
    st.rerun()
