from flask import Flask
import torch
import transformers
from flask import render_template, jsonify, redirect, url_for, request

app = Flask(__name__)

device = 'cpu'

@app.route('/')
@app.route('/home')
def home_page():
    return render_template('index.html')

@app.route('/storyteller', methods=['GET','POST'])
def storyteller_page():
    if request.method == 'POST':
        prompt = request.form.get("prompt")
        theme = request.form.get("theme")
        query = ""
        model = None
        tokenizer = None
        if theme == "horror":
          query = 'Generate a horror story given the beginning of the story: '+prompt
          model = transformers.AutoModelForCausalLM.from_pretrained("/content/drive/MyDrive/Story Generation Model Weights/horror_story/Gemma-Horror-Story-V1")
          tokenizer = transformers.AutoTokenizer.from_pretrained("/content/drive/MyDrive/Story Generation Model Weights/horror_story/Gemma-Horror-Story-V1")
          return render_template('storyteller.html', story=answer)
        elif theme == "adventure":
          query = 'Generate an adventure story given the beginning of the story: '+prompt
          model = transformers.AutoModelForCausalLM.from_pretrained("/content/drive/MyDrive/Story Generation Model Weights/adventure_story/Gemma-Adventure-Story-V1")
          tokenizer =  transformers.AutoTokenizer.from_pretrained("/content/drive/MyDrive/Story Generation Model Weights/adventure_story/Gemma-Adventure-Story-V1")
          return render_template('storyteller.html', story=answer)
        else:
          query = 'Generate a science friction story given the beginning of the story: '+prompt
          model = transformers.AutoModelForCausalLM.from_pretrained("/content/drive/MyDrive/Story Generation Model Weights/scifi_story/Gemma-Scifi-Story-V1")
          tokenizer = transformers.AutoTokenizer.from_pretrained("/content/drive/MyDrive/Story Generation Model Weights/scifi_story/Gemma-Scifi-Story-V1")
        prompt_template = """<start_of_turn>
        user
        Below is an instruction that describes a task. Write a response that appropriately completes the request.
        {query}
        <end_of_turn>\\n<start_of_turn> model

        """
        temp = """
        user
        Below is an instruction that describes a task. Write a response that appropriately completes the request.
        {query}
        \\n model

        """
        new_prompt = prompt_template.format(query=query)
        new_temp = temp.format(query=query)
        encoder = tokenizer(new_prompt, return_tensors="pt", add_special_tokens=True)
        model_inputs = encoder.to(device)
        generated_ids = model.generate(**model_inputs, max_new_tokens=500, do_sample=True, pad_token_id=tokenizer.eos_token_id)
        decoded = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        answer = (decoded)[len(new_temp):]
        return render_template('storyteller.html', story=answer)
    return render_template('storyteller.html')

@app.route('/team')
def team_page():
    return render_template('team.html')

if __name__ == '__main__':
    app.run(debug=True)
