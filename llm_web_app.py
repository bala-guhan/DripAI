import torch
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import streamlit as st

st.title("Drip AI")

model_name = "google-t5/t5-small"

# Load the model and the tokenizer from the saved directory
model_directory = "D:/projects/LLM integration in web"
model = AutoModelForSeq2SeqLM.from_pretrained(model_directory)
tokenizer = AutoTokenizer.from_pretrained(model_directory)

# Example text for the model (e.g., for translation or summarization)
# text = "Translate English to French: The weather is great today."
text_input = st.text_input("Enter your prompt in here in english: ")
text = f"Translate English to French: {text_input}"

if text:
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt")

    # Generate output, specifying max_new_tokens to control the length of the output
    outputs = model.generate(**inputs, max_new_tokens=50)  # Set the max number of tokens to generate

    # Decode the generated tokens back into text
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    st.write("Translation in French")
    st.code(decoded_output)
