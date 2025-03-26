from llama_cpp import Llama

# Load the GGUF model
llm = Llama(model_path="C:\\Users\\santhoj\\AppData\\Roaming\\Jan\\data\\models\\imported\\unsloth.Q4_K_M.gguf")

# Run inference
prompt = "I have trouble with my ecm in my excavator for my guidance solution."
output = llm(prompt, max_tokens=100)

print(output)
