from flask import Flask, request, jsonify
from llama_cpp import Llama
import warnings

warnings.filterwarnings("ignore")

# Initialize the Llama model (adjust path to your GGUF model)
llm = Llama(
    model_path="C:\\Users\\santhoj\\AppData\\Roaming\\Jan\\data\\models\\imported\\unsloth.Q4_K_M.gguf",  # Must be in GGUF format
    n_ctx=8192,                   
    n_threads=8,               
    n_gpu_layers=29,  
    verbose=False,            
)

app = Flask(__name__)

def generate_response(prompt, system_message="You are a helpful AI assistant."):
    formatted_prompt = f"""<|start_header_id|>system<|end_header_id|>

{system_message}<|eot_id|> <|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|> <|start_header_id|>assistant<|end_header_id|>

"""
    response = llm.create_completion(
        prompt=formatted_prompt,
        max_tokens=8192,           
        temperature=0.7,           
        top_p=0.95,                
        stop=["<|eot_id|>"],       
        echo=False,                
    )
    return response

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data.get('prompt', '')
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400
    
    try:
        output = generate_response(prompt)
        return jsonify({'output': output})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
