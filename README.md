# Patient‑Friendly Summarization of Clinical Discharge Notes (Streamlit + RAG)

A lightweight system that turns complex clinical discharge notes into clear, patient‑friendly summaries. It combines fine‑tuned seq2seq models (e.g., T5/BART) with an optional Retrieval‑Augmented Generation (RAG) step that pulls lay definitions for medical jargon.

---

## ✨ Features

- **Abstractive summarization** using fine‑tuned transformer models (e.g., `t5-small`, `facebook/bart-base`).
- **Optional RAG layer**: retrieves lay explanations for medical terms.
- **Metrics**: ROUGE, BLEU, BERTScore, optional readability (FKGL).
- **Streamlit UI** for quick demos, side‑by‑side comparisons, and exporting results.

---

## ⚙️ Requirements

- Python 3.9+
- pip/venv (or conda) and (optionally) a GPU-enabled PyTorch install

**Minimal `requirements.txt` (example):**

```
transformers>=4.42.0
datasets
evaluate
rouge-score
bert-score
sentence-transformers
faiss-cpu
streamlit
pandas
numpy
scikit-learn
nltk
python-dotenv
torch
```

Install:

```bash
python -m venv .venv && source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"
```

---

```

**Lay‑term CSV format**

```

term,lay_definition
hypertension,High blood pressure...
myocardial infarction,Heart attack...

````

You can start with ~100 common terms and expand over time.

---

## 🚀 Run the App

From the project root:

```bash
streamlit run app.py
````

**What you’ll see:**

1. Paste a clinical discharge note in the text area.
2. Choose a model (e.g., T5 or BART).
3. (Optional) Toggle RAG for jargon explanations.
4. Click **Summarize** to see a patient‑friendly summary and a table of term explanations.

---

## 🧠 Notes on Data

- We work with de‑identified clinical notes (e.g., MIMIC‑IV). Follow all data access rules and do **not** include PHI.
- For patient‑facing simplification, pair your notes with lay‑term definitions to power the RAG explanations.
- If you don’t have a large lay corpus yet, start small and iterate (quality > size).

---

## 📈 Expected Results & Limitations

- Fine‑tuning typically improves ROUGE/BERTScore and readability vs. base checkpoints.
- RAG adds **factual grounding** for medical terms so summaries are clearer for non‑experts.
- **Limitation:** With limited GPU, training steps/sequence length/batch size may be constrained. **With more GPU access we expect better results** (larger models, longer contexts, and broader hyperparameter sweeps).

---

## Acknowledgements

- Open‑source models and libraries from the Hugging Face ecosystem
- Clinical notes research community and publicly available datasets

---

## 💡 Quick Commands Recap

```bash
# 1) Install
pip install -r requirements.txt

# 2) Run Streamlit
streamlit run app.py
```
