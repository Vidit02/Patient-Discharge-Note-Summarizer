import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import textstat

MODEL_MAP = {
    "T5": "Vidit202/t5-mimic-summary",
    "BART": "Vidit202/bart-mimic-summary",
    "Pegasus": "Vidit202/pegasus-pubmed-summary"
}

@st.cache_resource
def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

@st.cache_resource
def load_retriever():
    med_dict = st.session_state.med_dict
    retriever = SentenceTransformer("all-MiniLM-L6-v2")
    terms = list(med_dict.keys())
    vectors = retriever.encode(terms, convert_to_numpy=True)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return retriever, index, terms, list(med_dict.values())

def generate_summary(text, model_name):
    tokenizer, model = load_model_and_tokenizer(MODEL_MAP[model_name])
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model.generate(**inputs, max_length=128)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def combine_summaries(sums):
    combined_input = " ".join(sums)
    return generate_summary(combined_input, "T5")

def simplify_rag(text):
    retriever, index, terms, lay_defs = load_retriever()
    tokens = text.split()
    simplified = []
    for word in tokens:
        cleaned = word.strip(",.?!:;()").lower()
        if cleaned in st.session_state.med_dict:
            simplified.append(st.session_state.med_dict[cleaned])
        else:
            emb = retriever.encode([cleaned], convert_to_numpy=True)
            D, I = index.search(emb, 1)
            if D[0][0] < 0.7:
                simplified.append(lay_defs[I[0][0]])
            else:
                simplified.append(word)
    return " ".join(simplified)

st.set_page_config(page_title="Clinical Note Simplifier", layout="wide")
st.title("🩺 Patient-Friendly Clinical Summary Generator")

default_dict = {
    "hypertension": "high blood pressure",
    "hypotension": "low blood pressure",
    "myocardial infarction": "heart attack",
    "cerebrovascular accident": "stroke",
    "dyspnea": "shortness of breath",
    "analgesic": "painkiller",
    "edema": "swelling",
    "febrile": "feverish",
    "gastritis": "stomach inflammation",
    "neoplasm": "tumor",
    "hyperlipidemia": "high cholesterol",
    "renal failure": "kidney failure",
    "hematemesis": "vomiting blood",
    "hematuria": "blood in urine",
    "hematochezia": "blood in stool",
    "tachycardia": "fast heartbeat",
    "bradycardia": "slow heartbeat",
    "arrhythmia": "irregular heartbeat",
    "diaphoresis": "sweating",
    "syncope": "fainting",
    "nausea": "feeling like vomiting",
    "vomitus": "vomit",
    "anemia": "low red blood cells",
    "leukocytosis": "high white blood cells",
    "thrombocytopenia": "low platelets",
    "polyuria": "frequent urination",
    "nocturia": "urinating at night",
    "dysuria": "painful urination",
    "incontinence": "loss of bladder control",
    "hepatomegaly": "enlarged liver",
    "splenomegaly": "enlarged spleen",
    "hepatitis": "liver inflammation",
    "cirrhosis": "liver scarring",
    "osteoporosis": "weak bones",
    "osteoarthritis": "joint wear and tear",
    "rheumatoid arthritis": "joint inflammation",
    "embolism": "blood clot",
    "thrombosis": "blood clot formation",
    "ischemia": "lack of blood flow",
    "necrosis": "tissue death",
    "cyanosis": "bluish skin",
    "jaundice": "yellow skin",
    "pruritus": "itching",
    "urticaria": "hives",
    "eczema": "skin inflammation",
    "dermatitis": "skin irritation",
    "alopecia": "hair loss",
    "melena": "black stool",
    "hemoptysis": "coughing up blood",
    "pleurisy": "lung lining inflammation",
    "pneumonia": "lung infection",
    "bronchitis": "airway inflammation",
    "asthma": "narrowed airways",
    "copd": "chronic lung disease",
    "emphysema": "damaged lung sacs",
    "hypoxia": "low oxygen",
    "apnea": "not breathing",
    "dyspnea on exertion": "shortness of breath with activity",
    "orthopnea": "breathing difficulty when lying down",
    "cyanotic": "blue-colored skin",
    "tachypnea": "fast breathing",
    "bradypnea": "slow breathing",
    "sepsis": "body-wide infection",
    "bacteremia": "bacteria in the blood",
    "pyelonephritis": "kidney infection",
    "cystitis": "bladder infection",
    "nephrolithiasis": "kidney stones",
    "cholelithiasis": "gallstones",
    "cholecystitis": "gallbladder inflammation",
    "pancreatitis": "pancreas inflammation",
    "appendicitis": "appendix inflammation",
    "diverticulitis": "colon pouch inflammation",
    "gastroenteritis": "stomach flu",
    "colitis": "colon inflammation",
    "constipation": "hard stool",
    "diarrhea": "loose stool",
    "abdominal distension": "bloated belly",
    "ascites": "fluid in belly",
    "anorexia": "loss of appetite",
    "cachexia": "wasting away",
    "obesity": "excess body weight",
    "malnutrition": "poor nutrition",
    "dehydration": "lack of fluids",
    "hypoglycemia": "low blood sugar",
    "hyperglycemia": "high blood sugar",
    "diabetes mellitus": "high blood sugar condition",
    "insulin resistance": "body not responding to insulin",
    "hypothyroidism": "low thyroid activity",
    "hyperthyroidism": "overactive thyroid",
    "goiter": "swollen thyroid",
    "dysphagia": "difficulty swallowing",
    "dysphasia": "difficulty speaking",
    "aphasia": "loss of ability to speak",
    "ataxia": "loss of coordination",
    "paresthesia": "tingling or numbness",
    "paralysis": "loss of movement",
    "spasticity": "stiff muscles",
    "seizure": "uncontrolled brain activity",
    "epilepsy": "recurring seizures",
    "headache": "head pain",
    "migraine": "intense headache",
    "encephalopathy": "brain dysfunction",
    "meningitis": "brain lining infection",
    "delirium": "confused thinking",
    "dementia": "memory loss",
    "psychosis": "loss of reality",
    "mania": "extreme mood elevation",
    "depression": "persistent sadness",
    "anxiety": "excessive worry",
    "hallucination": "seeing or hearing things",
    "delusion": "false belief",
    "bipolar disorder": "mood swings",
    "schizophrenia": "chronic mental illness",
    "ptsd": "trauma-related stress",
    "ocd": "repetitive thoughts or actions",
    "insomnia": "difficulty sleeping",
    "narcolepsy": "excessive daytime sleepiness",
    "valvular disease": "heart valve problem",
    "congestive heart failure": "heart pumping problem",
    "cardiomyopathy": "heart muscle disease",
    "pericarditis": "heart lining inflammation",
    "angina": "chest pain",
    "aortic aneurysm": "bulge in the aorta",
    "deep vein thrombosis": "blood clot in leg vein",
    "pulmonary embolism": "clot in lung artery",
    "shock": "dangerously low blood pressure",
    "anaphylaxis": "severe allergic reaction",
    "psoriasis": "scaly skin",
    "seborrheic dermatitis": "flaky scalp",
    "keratosis": "skin bump",
    "melanoma": "skin cancer",
    "basal cell carcinoma": "skin cancer",
    "squamous cell carcinoma": "skin cancer",
    "biopsy": "tissue sample",
    "pathology": "study of disease",
    "benign": "not cancerous",
    "malignant": "cancerous",
    "metastasis": "spread of cancer",
    "oncology": "cancer care",
    "radiotherapy": "radiation treatment",
    "chemotherapy": "cancer drug treatment",
    "immunotherapy": "immune-based treatment",
    "lymphadenopathy": "swollen lymph nodes",
    "tonsillitis": "tonsil infection",
    "sinusitis": "sinus infection",
    "otitis media": "ear infection",
    "conjunctivitis": "pink eye",
    "pharyngitis": "sore throat",
    "laryngitis": "voice box inflammation",
    "bronchiolitis": "small airway inflammation",
    "pneumothorax": "collapsed lung",
    "pleural effusion": "fluid around lungs",
    "atelectasis": "lung collapse",
    "intubation": "inserting breathing tube",
    "extubation": "removing breathing tube",
    "tracheostomy": "neck breathing tube",
    "ventilator": "breathing machine",
    "resuscitation": "reviving from death",
    "cardiac arrest": "heart stops",
    "do not resuscitate": "no revival order",
    "code blue": "medical emergency",
    "advance directive": "care instructions in advance",
    "palliative care": "comfort care",
    "hospice": "end-of-life care",
    "autopsy": "after-death exam",
    "morbidity": "illness",
    "mortality": "death",
    "prognosis": "expected outcome",
    "diagnosis": "identified condition",
    "treatment": "care plan",
    "prescription": "medicine order",
    "dosage": "medicine amount",
    "contraindication": "reason not to use",
    "adverse reaction": "bad effect",
    "side effect": "unintended effect",
    "tolerance": "resistance to drug",
    "dependence": "reliance on drug",
    "withdrawal": "symptoms after stopping",
    "overdose": "too much medicine",
    "toxicity": "poison effect",
    "placebo": "fake treatment",
    "clinical trial": "research study",
    "vital signs": "body measurements",
    "temperature": "body heat",
    "pulse": "heartbeat rate",
    "respiration": "breathing rate",
    "blood pressure": "pressure in arteries"
}

st.session_state.med_dict = default_dict

note_input = st.text_area("📄 Enter Clinical Discharge Note", height=200)
col1, _ = st.columns([2, 1])
with col1:
    selected_model = st.selectbox("Choose Model", ["T5", "BART", "Pegasus", "Combined"])

submit = st.button("Generate Summary")

if submit and note_input.strip():
    with st.spinner("Generating summary..."):
        if selected_model == "Combined":
            s1 = generate_summary(note_input, "T5")
            s2 = generate_summary(note_input, "BART")
            s3 = generate_summary(note_input, "Pegasus")
            final_summary = combine_summaries([s1, s2, s3])
        else:
            final_summary = generate_summary(note_input, selected_model)

        simplified_summary = simplify_rag(final_summary)
        fkgl_score = textstat.flesch_kincaid_grade(simplified_summary)

    st.markdown("### 📝 Original Note")
    st.info(note_input)

    st.markdown("### 📌 Model Summary")
    st.success(final_summary)

    st.markdown("### 🧠 RAG-Based Simplified Summary")
    st.success(simplified_summary)

    st.markdown(f"**Readability (FKGL):** {fkgl_score:.2f}")
