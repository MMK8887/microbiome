An interactive AI-powered Streamlit app that helps users understand gut health through microbiome data, natural language queries, and personalized recommendations.

---

## 🔍 Features

- 🧬 Ask questions about gut health, diet, probiotics, and fiber
- 🗂️ NLP-driven intent detection using rule-based + ML backup
- 🧠 Named entity and keyword extraction from queries
- 📊 Upload microbiome CSV data and get diversity/risk scores
- 📈 Personalized dietary suggestions based on profile
- 🧠 Learns from user feedback using local JSON storage

---

## 📁 File Structure

```bash
microbiome_ai/
├── app.py                   # Main Streamlit app
├── learned_intents.json     # User-trained Q&A (optional)
├── requirements.txt         # Python packages
├── README.md                # Project overview
└── sample_data.csv          # Optional input example
