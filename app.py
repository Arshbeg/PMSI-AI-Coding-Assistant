import streamlit as st
import torch
import torch.nn as nn
import transformers
import numpy as np
import pickle
import os

st.set_page_config(page_title="PMSI AI Assistant", page_icon="🏥")

# configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "pmsi_model_high_conf.bin")
LABELS_PATH = os.path.join(BASE_DIR, "models", "mlb_classes.pkl")
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"

# setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# rsrcs
@st.cache_resource
def load_resources():
    # Load Labels
    with open(LABELS_PATH, 'rb') as f:
        classes = pickle.load(f)
        
    # tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    
    class PMSIModel(nn.Module):
        def __init__(self, n_classes):
            super(PMSIModel, self).__init__()
            self.bert = transformers.AutoModel.from_pretrained(MODEL_NAME)
            self.drop = nn.Dropout(0.3)
            self.out = nn.Linear(768, n_classes)

        def forward(self, ids, mask, token_type_ids):
            _, pooled_output = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
            output = self.drop(pooled_output)
            return self.out(output)
            
    # Load Weights
    model = PMSIModel(len(classes))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    
    return tokenizer, model, classes

# Load everything
try:
    tokenizer, model, classes = load_resources()
except FileNotFoundError:
    st.error("Model files not found. Check 'models' folder.")
    st.stop()

# hardcoding 10 keywords for demo
icd_10_map = {
    'hernia': 'K40 - Inguinal Hernia',
    'surgery': 'Z98.8 - Surgical Aftercare',
    'gastroenterology': 'K92.9 - Digestive Disease',
    'laparoscopic': 'Z90.8 - Laparoscopic Procedure',
    'abdominal': 'R10.9 - Unspecified Abdominal Pain',
    'cardiovascular / pulmonary': 'I51.9 - Heart Disease',
    'orthopedic': 'M25.9 - Joint Disorder',
    'fracture': 'S72 - Fracture of Femur',
    'neurology': 'G98 - Nervous System Disorder',
    'radiology': 'Z01.8 - Radiological Exam'
}

# prediction function
def predict(text):
    inputs = tokenizer.encode_plus(
        text, None, add_special_tokens=True, max_length=128,
        padding='max_length', truncation=True, return_token_type_ids=True
    )

    ids = torch.tensor(inputs['input_ids'], dtype=torch.long).unsqueeze(0).to(device)
    mask = torch.tensor(inputs['attention_mask'], dtype=torch.long).unsqueeze(0).to(device)
    token_type_ids = torch.tensor(inputs['token_type_ids'], dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(ids, mask, token_type_ids)
    
    probs = torch.sigmoid(outputs).cpu().numpy()[0]
    
    results = []
    for idx, score in enumerate(probs):
        label = classes[idx]
        display_name = icd_10_map.get(label, label.title())
        results.append((display_name, score))
        
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:5] # Return Top 5

st.title("🏥 PMSI Coding Assistant")
st.markdown("This AI reads medical reports and suggests **ICD-10 (CIM-10) Codes** automatically.")

# Input Area
text_input = st.text_area("Paste Medical Report Here:", height=150, 
                         value="Patient is a 45-year-old male presenting with severe abdominal pain and bloating. Scheduled for surgery.")

if st.button("Analyze Report"):
    with st.spinner("Analyzing text..."):
        predictions = predict(text_input)
        
    st.success("Analysis Complete!")
    
    st.subheader("Recommended Codes:")
    
    for name, score in predictions:
        if score > 0.01: # Filter out absolute zeroes
            st.divider()
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{name}**")
                st.progress(int(score*100))
            with col2:
                st.metric("Confidence", f"{score:.1%}")

# Sidebar info
st.sidebar.title("About")
st.sidebar.info("Built with Bio_ClinicalBERT.\nTrained on MTSamples Dataset.")