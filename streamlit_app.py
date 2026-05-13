import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==========================
# TITLE
# ==========================

st.title("SEC Investment Sentiment Analyzer")

st.write(
    "Upload a SEC filing and get AI-powered investment analysis."
)

# ==========================
# LOAD MODEL
# ==========================

@st.cache_resource
def load_model():

    model_name = "Qwen/Qwen2-0.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(model_name)

    return tokenizer, model

tokenizer, model = load_model()

# ==========================
# FILE UPLOAD
# ==========================

uploaded_file = st.file_uploader(
    "Upload SEC Filing (.txt)",
    type=["txt"]
)

# ==========================
# ANALYSIS
# ==========================

if uploaded_file is not None:

    text = uploaded_file.read().decode("utf-8")

    text = text[:3000]

    st.success("File uploaded successfully!")

    if st.button("Analyze Filing"):

        with st.spinner("Analyzing filing..."):

            prompt = f"""
You are an AI financial analyst.

Analyze this SEC filing.

Respond ONLY in this format:

Sentiment:
Risks:
Positive Indicators:
Investment Recommendation:

SEC Filing:
{text}
"""

            inputs = tokenizer(
                prompt,
                return_tensors="pt"
            )

            outputs = model.generate(
                **inputs,
                max_new_tokens=80
            )

            response = tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )

        st.subheader("AI Financial Analysis")

        st.write(response)