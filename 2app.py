import requests
from transformers import pipeline

# =====================================
# MODELO
# =====================================

pipe = pipeline(
    "text-generation",
    model="Qwen/Qwen2-0.5B-Instruct"
)

# =====================================
# DOCUMENTO DE PRUEBA
# =====================================

url = "https://www.gutenberg.org/files/1342/1342-0.txt"

response = requests.get(url)

text = response.text

# recortar texto
text = text[:3000]

# =====================================
# PROMPT FINANCIERO
# =====================================

prompt = f"""
You are a financial analyst.

Analyze the sentiment of this document.

Return ONLY:

1. Overall Sentiment
2. Risks
3. Positive Indicators
4. Investment Opinion

Text:
{text}
"""

# =====================================
# GENERAR
# =====================================

result = pipe(
    prompt,
    max_new_tokens=120,
    do_sample=False
)

# =====================================
# OUTPUT
# =====================================

print("\n====================")
print("AI INVESTMENT ANALYSIS")
print("====================\n")

print(result[0]["generated_text"])