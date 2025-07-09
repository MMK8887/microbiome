import streamlit as st
import pandas as pd
import os, json, nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag, ne_chunk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from datetime import datetime

st.set_page_config(page_title="MIOME (Gut Health Assistant)", page_icon="🤖")

# Download NLTK resources
for r, p in {
    'punkt': 'tokenizers/punkt', 'stopwords': 'corpora/stopwords',
    'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger',
    'maxent_ne_chunker': 'chunkers/maxent_ne_chunker', 'words': 'corpora/words'
}.items():
    try:
        nltk.data.find(p)
    except LookupError:
        nltk.download(r)

stop_words = set(stopwords.words("english"))

@st.cache_data
def load_microbiome_data():
    try:
        df = pd.read_csv("microbiome_data.csv")
        df["date"] = pd.to_datetime(df["date"], errors='coerce')
        return df
    except FileNotFoundError:
        return pd.DataFrame(columns=["user", "date", "fiber_score", "probiotic_score", "diversity_score", "recommendation"])

micro_df = load_microbiome_data()

def detect_intent(text):
    t = text.lower()
    if any(x in t for x in ["hello", "hi", "hey", "yo", "good morning", "good evening"]): return "hello"
    if any(x in t for x in ["gut", "digestion", "microbiome"]): return "gut_health"
    if any(x in t for x in ["probiotic", "fermented"]): return "probiotic_info"
    if any(x in t for x in ["fiber", "prebiotic"]): return "fiber_info"
    if any(x in t for x in ["shannon", "diversity", "risk"]): return "diversity_info"
    if any(x in t for x in ["diet", "food", "meal", "eat"]): return "diet_suggestion"
    if any(x in t for x in ["help", "what", "do", "can"]): return "help"
    return "unknown"

training_path = "training_data.csv"
if os.path.exists(training_path):
    df_train = pd.read_csv(training_path)
    texts = df_train["text"].tolist()
    labels = df_train["intent"].tolist()
else:
    texts = ["Tell me about probiotics", "gut pain", "food to eat", "kimchi", "Shannon index", "hello", "hi"]
    labels = ["probiotic_info", "gut_health", "diet_suggestion", "probiotic_info", "diversity_info", "hello", "hello"]

vec = TfidfVectorizer()
X_train = vec.fit_transform(texts)
clf = LogisticRegression().fit(X_train, labels)

def ml_detect_intent(text, threshold=0.3):
    X_test = vec.transform([text])
    proba = clf.predict_proba(X_test).max()
    return clf.predict(X_test)[0] if proba >= threshold else "unknown"

learn_path = "learned_intents.json"
learned = json.load(open(learn_path)) if os.path.exists(learn_path) else {}

def learnable_detect(text):
    key = text.lower().strip()
    val = learned.get(key)
    if isinstance(val, dict): return val
    elif isinstance(val, str): return {"intent": val}
    else: return {"intent": detect_intent(text)}

def update_learning(q, custom_response=None, intent_label=None):
    key = q.lower().strip()
    if custom_response:
        learned[key] = {"intent": "custom", "response": custom_response}
    elif intent_label:
        learned[key] = {"intent": intent_label}
    with open(learn_path, "w") as f:
        json.dump(learned, f, indent=2)

def train_recommendation_model(df):
    if df.empty or df.shape[0] < 3:
        return None
    X = df[["fiber_score", "probiotic_score", "diversity_score"]]
    y = df["recommendation"]
    model = DecisionTreeClassifier(max_depth=3)
    return model.fit(X, y)

rec_model = train_recommendation_model(micro_df)
latest, recommendation, insights = None, None, []

st.title("🤖 MIOME: Your Gut Health Assistant")
st.markdown("💬 Ask anything about **gut health**, **probiotics**, **fiber**, **diet**, or **microbiome diversity**.")

user = st.sidebar.text_input("👤 Enter your name", value="Guest")
uploaded_file = st.sidebar.file_uploader("📄 Upload Microbiome Data (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".xlsx"):
            new_data = pd.read_excel(uploaded_file)
        else:
            new_data = pd.read_csv(uploaded_file)
    except Exception:
        st.sidebar.error("❗ Failed to read file.")
        new_data = pd.DataFrame()

    if {"fiber_score", "probiotic_score", "diversity_score"}.issubset(new_data.columns):
        new_data["user"] = user
        new_data["date"] = pd.to_datetime(datetime.now().strftime("%Y-%m-%d"))
        micro_df = pd.concat([micro_df, new_data], ignore_index=True)
        micro_df.to_csv("microbiome_data.csv", index=False)
        rec_model = train_recommendation_model(micro_df)
        st.sidebar.success("✅ Uploaded and saved!")

        latest = new_data.iloc[-1]
        X_new = latest[["fiber_score", "probiotic_score", "diversity_score"]].values.reshape(1, -1)
        recommendation = rec_model.predict(X_new)[0] if rec_model else None

        insights = []
        if latest["fiber_score"] < 5:
            insights.append("Low fiber → eat more oats, legumes, fruits.")
        if latest["probiotic_score"] < 5:
            insights.append("Low probiotics → try yogurt, kefir, kimchi.")
        if latest["diversity_score"] < 5:
            insights.append("Low diversity → eat a variety of vegetables.")

        st.markdown(f"""
        ## 📋 Report Summary  
        **Date:** {latest['date'].date()}  
        **Recommendation:** 🟢 **{recommendation}**  
        **Insights:** {', '.join(insights) if insights else "✅ All scores look healthy!"}
        """)

        st.sidebar.metric("Fiber Score", latest["fiber_score"])
        st.sidebar.metric("Probiotic Score", latest["probiotic_score"])
        st.sidebar.metric("Diversity Score", latest["diversity_score"])
        chart_df = pd.DataFrame({
            "Metric": ["Fiber", "Probiotic", "Diversity"],
            "Score": [latest["fiber_score"], latest["probiotic_score"], latest["diversity_score"]]
        })
        st.sidebar.bar_chart(chart_df.set_index("Metric"))
    else:
        st.sidebar.error("❗ Missing required columns.")

def generate_response(intent_obj, user=None):
    global latest, recommendation, insights
    responses = {
        "hello": "👋 Hello! I'm your gut health assistant.",
        "gut_health": "🧬 The gut microbiome supports digestion, immunity, and mood regulation.",
        "probiotic_info": "🧪 Probiotics are beneficial microbes in yogurt, kefir, kimchi, and supplements.",
        "fiber_info": "🌾 Fiber feeds your gut bacteria. Eat fruits, veggies, legumes, and oats.",
        "diversity_info": "🧫 A diverse microbiome is healthiest. Add variety to your meals!",
        "diet_suggestion": "🥗 Include fermented foods, leafy greens, and avoid added sugars.",
        "help": "🧠 You can ask about gut health, fiber, diet, or upload your data for analysis.",
        "unknown": "🤔 I'm still learning. I haven't seen that question before. You can help me improve by giving feedback below!"
    }
    if isinstance(intent_obj, dict) and "response" in intent_obj:
        return intent_obj["response"]
    intent = intent_obj.get("intent", "unknown")
    if intent == "diet_suggestion" and user:
        user_data = micro_df[micro_df["user"] == user]
        if not user_data.empty:
            latest = user_data.sort_values("date", ascending=False).iloc[0]
            insights = []
            if latest["fiber_score"] < 5:
                insights.append("Low fiber → try oats, fruits, legumes")
            if latest["probiotic_score"] < 5:
                insights.append("Low probiotics → try yogurt or kimchi")
            if latest["diversity_score"] < 5:
                insights.append("Low diversity → include a variety of veggies")
            recommendation = latest["recommendation"]
            return f"🍽️ Based on your scores: **{recommendation}**\n\n📌 Suggestions: {', '.join(insights)}"
    return responses.get(intent, responses["unknown"])

# --- Chat State ---
if "history" not in st.session_state:
    st.session_state.history = []
if "is_unknown" not in st.session_state:
    st.session_state.is_unknown = False
if "show_correction_box" not in st.session_state:
    st.session_state.show_correction_box = False

# --- Chat Input ---
user_input = st.chat_input("Type your question...")

if user_input:
    intent_obj = learnable_detect(user_input)
    st.session_state.is_unknown = intent_obj.get("intent") == "unknown"
    if st.session_state.is_unknown:
        intent_obj["intent"] = ml_detect_intent(user_input)
        update_learning(user_input, intent_label="unknown")
    bot_reply = generate_response(intent_obj, user)
    st.session_state.history.append(("You", user_input))
    st.session_state.history.append(("Bot", bot_reply))
    st.session_state.show_correction_box = False

# --- Chat Display ---
for speaker, msg in st.session_state.history:
    st.markdown(f"**{speaker}:** {msg}")

# --- Feedback Section ---
if st.session_state.history and st.session_state.history[-1][0] == "Bot":
    st.markdown("📣 Was this helpful?")
    col1, col2, _ = st.columns([0.05, 0.05, 0.9])

    if col1.button("👍"):
        st.success("Thanks for the feedback!")
        if st.session_state.is_unknown:
            update_learning(st.session_state.history[-2][1], intent_label=ml_detect_intent(st.session_state.history[-2][1]))
        st.session_state.show_correction_box = False

    if col2.button("👎"):
        if st.session_state.show_correction_box:
            # Second 👎 pressed without input — remove bot message
            if st.session_state.history[-1][0] == "Bot":
                st.session_state.history.pop()
            st.session_state.show_correction_box = False
        else:
            st.session_state.show_correction_box = True

# --- Correction Box ---
if st.session_state.show_correction_box:
    corrected = st.text_input("Suggest a better response:")
    if corrected:
        update_learning(st.session_state.history[-2][1], custom_response=corrected)
        st.success("✅ Thanks! I’ll remember that.")
        st.session_state.show_correction_box = False

# --- Score History ---
with st.sidebar.expander("📈 My Full Score History"):
    user_data = micro_df[micro_df["user"] == user]
    if not user_data.empty:
        user_data = user_data.sort_values("date")
        st.dataframe(user_data.sort_values("date", ascending=False))
        st.line_chart(user_data.set_index("date")[['fiber_score', 'probiotic_score', 'diversity_score']])
    else:
        st.write("No data available.")
