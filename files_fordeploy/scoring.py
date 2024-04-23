import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from azureml.core.model import Model

def init():
    global model, tokenizer
    model_path = Model.get_model_path('Story Generation Model Weights/horror_story/Gemma-Horror-Story-V1')
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

def run(raw_data):
    data = json.loads(raw_data)
    prompt = data['prompt']
    theme = data['theme']

    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=500, num_return_sequences=1)
    story = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return json.dumps({"story": story})
