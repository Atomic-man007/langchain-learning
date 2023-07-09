
import torch
from langchain import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM

# Check if cuda is available 
torch.cuda.is_available()

model_id = "tiiuae/falcon-7b-instruct" #tiiuae/falcon-40b-instruct

tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load Model 
model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir='./workspace/', 
    torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto", offload_folder="offload")

# Set PT model to inference mode
model.eval()
# Build HF Transformers pipeline 
pipeline = pipeline(
    "text-generation", 
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    max_length=400,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
)

llm = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {'temperature':0})

from langchain import PromptTemplate,  LLMChain

template = """
You are an intelligent chatbot. Help the following question with brilliant answers.
Question: {question}
Answer:"""
prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(prompt=prompt, llm=llm)


# Import gradio for UI
import gradio as gr
# Create generate function - this will be called when a user runs the gradio app 
def generate(prompt): 
    # The prompt will get passed to the LLM Chain!
    return llm_chain.run(prompt)
    # And will return responses 
    
# Define a string variable to hold the title of the app
title = '🦜🔗 Falcon-7b-Instruct'
# Define another string variable to hold the description of the app
description = 'This application demonstrates the use of the open-source `Falcon-7b-Instruct` LLM.'
# pls subscribe 🙏
# Build gradio interface, define inputs and outputs...just text in this
gr.Interface(fn=generate, inputs=["text"], outputs=["text"], 
             # Pass through title and description
             title=title, description=description, 
             # Set theme and launch parameters
             theme='derekzen/stardust').launch(server_port=8080, share=True)