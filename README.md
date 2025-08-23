# 📖 MIOME — Gut Health Assistant

MIOME is an interactive **Streamlit web app** that helps users **analyze and improve their gut health**.  
It combines **data analysis**, **lightweight ML/NLP for chatbot intent detection**, and **personalized feedback** into a single assistant.

---

## 🔑 Key Features

### 📂 Data Upload & Processing
- Upload **CSV/XLSX files** with gut health metrics:
  - `fiber_score`, `probiotic_score`, `diversity_score` (0–10 scale).
- Cleans, validates, and clips scores into a safe range (0–10).
- Supports an optional **date column** for time-series tracking.

### 📊 Gut Health Report
- Computes an **overall Gut Score**:  
  ```
  Gut Score = 0.4 × fiber + 0.3 × probiotic + 0.3 × diversity
  ```
- Categorizes into zones:
  - 🟢 Green (healthy)
  - 🟡 Yellow (moderate)
  - 🔴 Red (needs attention)
- Displays metrics dashboard with individual scores.
- Generates **rule-based personalized tips**.

### 🤖 Chatbot (MIOME Bot)
- **NLP-powered intent detection** using:
  - ✅ SentenceTransformers (`all-MiniLM-L6-v2`) if available
  - ⚡ TF-IDF + cosine similarity as fallback
- Recognizes user intents:
  - Greetings, Report Summary, Diet, Improvements  
  - Fiber, Probiotic, Diversity info  
  - Rainbow Plate, Help, Thanks, Goodbye
- Provides **contextual, emoji-rich responses**.
- Supports **follow-up conversations** (e.g., `"yes"` → summary).

### 📘 Knowledge Base
- Explains **probiotic foods**: kefir, kimchi, miso, sauerkraut, tempeh.
- Explains gut health concepts like **“Rainbow Plate”**.

### 💬 Interactive Chat UI
- Persistent chat history.
- **Clear chat** option in sidebar.
- User-friendly conversational interface.

---

## ⚙️ Tech Stack
- **Frontend/UI**: Streamlit  
- **Data Processing**: pandas, numpy  
- **ML/NLP**:
  - sentence-transformers (MiniLM embeddings)  
  - scikit-learn (TF-IDF, cosine similarity fallback)  
- **Visualization**: Streamlit (`st.metric`, `st.info`, `st.success`)  

---

## 🧠 Example Workflow

### 1. Upload gut health file:
```csv
date,fiber_score,probiotic_score,diversity_score
2025-08-01,6,5,7
2025-08-10,7,6,8
```

### 2. App shows:
- 🌾 Fiber: 7.0/10 ✅  
- 🦠 Probiotic: 6.0/10 ⚠️  
- 🥗 Diversity: 8.0/10 ✅  
- ➡️ Gut Score: **7.1/10 🟢 Green Zone**  

💡 Personalized Tip:  
🦠 *Add probiotics: yogurt, kefir, kimchi, sauerkraut, kombucha.*

### 3. Chat with MIOME:
**User:** `Show me my summary`  
**Bot:**  
```
📊 Report Summary:
- 🌾 Fiber: 7.0/10
- 🦠 Probiotic: 6.0/10
- 🥗 Diversity: 8.0/10
➡️ Gut Score: 7.1/10 (Green zone)

💡 Tip: 🦠 Add probiotics: yogurt, kefir, kimchi, sauerkraut, kombucha.
```

---

## 🚀 Getting Started

1. Clone this repo:
   ```bash
   git clone https://github.com/your-username/miome.git
   cd miome
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

---

## 📌 Roadmap
- [ ] Add user authentication  
- [ ] Improve chatbot with memory & context-awareness  
- [ ] Expand knowledge base with diet & lifestyle tips  
- [ ] Deploy to Streamlit Cloud / HuggingFace Spaces  

---

## 🙌 Acknowledgments
- [Streamlit](https://streamlit.io/) for the simple UI framework  
- [SentenceTransformers](https://www.sbert.net/) for NLP embeddings  
- [scikit-learn](https://scikit-learn.org/) for fallback intent detection  

---
>DEVELOPER
MANISH M KUMAR
