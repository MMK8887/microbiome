import streamlit as st
import pandas as pd, numpy as np, json, os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag, ne_chunk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# === Download NLTK Resources ===
for r, p in {
    'punkt': 'tokenizers/punkt',
    'stopwords': 'corpora/stopwords',
    'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger',
    'maxent_ne_chunker': 'chunkers/maxent_ne_chunker',
    'words': 'corpora/words'
}.items():
    try: nltk.data.find(p)
    except LookupError: nltk.download(r)

# === NLP Utilities ===
stop_words = set(stopwords.words("english"))

def extract_keywords(text):
    tokens = word_tokenize(text)
    return [w for w, p in pos_tag([w for w in tokens if w.isalnum() and w.lower() not in stop_words]) if p.startswith("NN") or p.startswith("JJ")]

def extract_named_entities(text):
    return [(" ".join([leaf[0] for leaf in tree.leaves()]), tree.label()) for tree in ne_chunk(pos_tag(word_tokenize(text))) if hasattr(tree, 'label')]

def detect_intent(text):
    t = text.lower()
    if any(x in t for x in ["gut", "digestion", "microbiome"]): return "gut_health"
    if any(x in t for x in ["probiotic", "fermented"]): return "probiotic_info"
    if any(x in t for x in ["fiber", "prebiotic"]): return "fiber_info"
    if any(x in t for x in ["shannon", "diversity", "risk"]): return "diversity_info"
    if any(x in t for x in ["diet", "food", "meal", "eat"]): return "diet_suggestion"
    if any(x in t for x in ["hello", "hi", "hey"]): return "hello"
    return "unknown"

responses = {
    "gut_health": "🧬 The gut microbiome supports digestion, immunity, and mental health.",
    "probiotic_info": "🧫 Probiotics are beneficial microbes found in yogurt, kefir, kimchi, and supplements.",
    "fiber_info": "🌾 Fiber feeds your good gut bacteria. Eat fruits, veggies, legumes, and oats.",
    "diversity_info": "🌈 A diverse microbiome is a healthy one. Variety in your diet helps!",
    "diet_suggestion": "🥗 Eat fermented foods, leafy greens, and avoid excess sugar for gut health.",
    "hello": "👋 Hello! I'm your gut health assistant.",
    "unknown": "❓ I'm still learning. Try asking about gut health, probiotics, fiber, or diet."
}

def generate_response(intent_obj):
    if isinstance(intent_obj, dict) and "response" in intent_obj:
        return intent_obj["response"]
    intent = intent_obj.get("intent", "unknown")
    return responses.get(intent, responses["unknown"])

# === ML Intent Classifier ===
texts = ["Tell me about probiotics", "digestion problems", "what to eat", "kimchi good", "Shannon index", "hi", "hello"]
labels = ["probiotic_info", "gut_health", "diet_suggestion", "probiotic_info", "diversity_info", "hello", "hello"]
vec = TfidfVectorizer()
clf = LogisticRegression().fit(vec.fit_transform(texts), labels)
def ml_detect_intent(text): return clf.predict(vec.transform([text]))[0]

# === Learnable Intent Store ===
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
    with open(learn_path, "w") as f: json.dump(learned, f, indent=2)
    st.success(f"✅ Updated intent for '{key}'")

# === Microbiome Analysis Utils ===
normalize_abundance = lambda df: df.div(df.sum(axis=1), axis=0)
compute_shannon_index = lambda df: -np.sum(df * np.log2(df + 1e-9), axis=1)
def compute_risk_score(s): return 1 - s

def recommend(df, p):
    out = []
    if "Bifidobacterium" in df.columns and df["Bifidobacterium"].mean() < 0.1:
        out.append("Try non-dairy kimchi or kombucha." if "dairy" in p["allergies"] else "Include yogurt or kefir.")
    if "Firmicutes" in df.columns and df["Firmicutes"].mean() > 0.4:
        out.append("Reduce processed sugars.")
    if p["diet"] == "vegetarian": out.append("Eat lentils, oats, leafy greens.")
    if "digestion" in p["goal"]: out.append("Add ginger, turmeric to your diet.")
    return out or ["You're doing great! Keep eating diverse foods."]

# === Streamlit App ===
st.set_page_config(page_title="Microbiome AI", layout="centered")
st.title("🧠 Microbiome AI Assistant")

# === Chatbot Section ===
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if q := st.chat_input("Ask me anything about gut health..."):
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    intent_obj = learnable_detect(q)
    if intent_obj["intent"] == "unknown":
        intent_obj = {"intent": ml_detect_intent(q)}

    response = generate_response(intent_obj)
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)

# === Microbiome Data Analysis ===
st.header("🧪 Microbiome Data Analysis")

uploaded = st.file_uploader("📂 Upload your microbiome CSV file", type="csv")

if uploaded:
    df = pd.read_csv(uploaded)
    sample_ids = df["SampleID"] if "SampleID" in df.columns else None
    if "SampleID" in df.columns: df.drop("SampleID", axis=1, inplace=True)
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

    df_norm = normalize_abundance(df)
    df_norm["Shannon_Index"] = compute_shannon_index(df_norm)
    df_norm["Risk_Score"] = compute_risk_score(df_norm["Shannon_Index"])
    if sample_ids is not None: df_norm.insert(0, "SampleID", sample_ids)

    st.subheader("📋 Normalized Data")
    st.dataframe(df_norm.round(4))

    st.subheader("📊 Summary Statistics")
    st.write(df_norm[["Shannon_Index", "Risk_Score"]].describe().round(3))

    st.subheader("🌈 Shannon Diversity Index")
    st.bar_chart(df_norm["Shannon_Index"])

    st.subheader("⚠️ Risk Score")
    st.line_chart(df_norm["Risk_Score"])

    st.subheader("🥗 Dietary Recommendations")
    profile = {"diet": "vegetarian", "allergies": ["dairy"], "goal": "improve digestion"}
    for r in recommend(df, profile):
        st.markdown(f"- {r}")
