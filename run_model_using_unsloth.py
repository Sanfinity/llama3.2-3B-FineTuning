from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="C:\\Users\\santhoj\\AppData\\Roaming\\Jan\\data\\models\\imported\\unsloth.Q4_K_M.yml",  # YAML-based fine-tuning setup
    max_seq_length=4096,
    dtype=None  # Automatically selects the best dtype
)

prompt = "Tell me a joke!"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
