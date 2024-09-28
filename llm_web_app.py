import torch
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os
import streamlit as st

st.title("Drip AI")

model_name = "google/t5-small"

# Path where the model and tokenizer are saved
model_directory = "./"  # Assuming the files are in the same directory as the app.py

# Load the model and tokenizer from the saved directory
model = AutoModelForSeq2SeqLM.from_pretrained(model_directory)
tokenizer = AutoTokenizer.from_pretrained(model_directory)

# Text input from user
text_input = st.text_input("Enter your prompt in English: ")
text = f"Translate English to French: {text_input}"

if text_input:
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt")

    # Generate output
    outputs = model.generate(**inputs, max_new_tokens=50)

    # Decode the generated tokens back into text
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Display the translation
    st.write("Translation in French:")
    st.code(decoded_output)
