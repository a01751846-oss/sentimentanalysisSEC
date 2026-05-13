from transformers import AutoTokenizer, AutoModelForCausalLM

# ==========================
# LOAD MODEL
# ==========================

model_name = "Qwen/Qwen2-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(model_name)

# ==========================
# FILE INPUT
# ==========================

file_path = input("\nEnter SEC file path (.txt): ")

with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()

# ==========================
# LIMIT TEXT SIZE
# ==========================

text = text[:3000]

# ==========================
# PROMPT
# ==========================

prompt = f"""
You are an AI financial analyst.

Analyze the following SEC filing.

Respond ONLY in this format:

Sentiment:
Risks:
Positive Indicators:
Investment Recommendation:

SEC Filing:
{text}
"""

# ==========================
# TOKENIZE
# ==========================

inputs = tokenizer(prompt, return_tensors="pt")

# ==========================
# GENERATE
# ==========================

outputs = model.generate(
    **inputs,
    max_new_tokens=80
)

# ==========================
# DECODE
# ==========================

response = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

# ==========================
# OUTPUT
# ==========================

print("\n====================")
print("SEC ANALYSIS")
print("====================\n")

print(response)