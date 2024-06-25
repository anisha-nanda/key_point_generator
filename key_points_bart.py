import streamlit as st
import time
import fitz
from transformers import pipeline, GPTNeoForCausalLM, GPT2Tokenizer
import torch

# streamlit run key_points_bart.py
# xcode-select --install
# pip install watchdog

# Streamlit page configuration
st.set_page_config(page_title='Key Points Generator', page_icon='🏗️', layout='wide')

# Function to read PDF
def read_pdf(file):
    pdf_document = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in pdf_document:
        text += page.get_text()
    return text

# Initialize models
@st.cache_resource
def load_models():
    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    except Exception as e:
        st.error(f"Error loading summarizer: {str(e)}")
        summarizer = None

    model_name = "EleutherAI/gpt-neo-125M"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPTNeoForCausalLM.from_pretrained(model_name).half()

    if torch.cuda.is_available():
        model = model.to('cuda')

    return summarizer, model, tokenizer

summarizer, model, tokenizer = load_models()

# Function to summarize text
def summarize_text(text):
    if summarizer is None:
        return text
    chunks = [text[i:i+1024] for i in range(0, len(text), 1024)]
    summaries = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=150, min_length=40, do_sample=False)
        summaries.append(summary[0]['summary_text'])
    return " ".join(summaries)

# Function to extract key points
def extract_key_points(prompt, document):
    inputs = tokenizer(prompt + document, return_tensors="pt", max_length=512, truncation=True)
    if torch.cuda.is_available():
        inputs = inputs.to('cuda')
    outputs = model.generate(**inputs, max_length=1000, num_return_sequences=1)
    return tokenizer.decode(outputs[0])


# Streamlit UI
st.title('Key Points Generator')

with st.expander('About this app'):
    st.info('This app helps users extract key points from a given document.')
    st.warning('To use the app, upload a PDF file in the sidebar.')

# Sidebar
with st.sidebar:
    st.header('Input data')
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    with st.status("Processing...", expanded=True) as status:
        try:
            document = read_pdf(uploaded_file)
            status.update(label="Summarizing...", state="running")
            summary = summarize_text(document)
            status.update(label="Extracting key points...", state="running")
            prompt = "Extract the key points from the following text:\n\n"
            key_points = extract_key_points(prompt, summary)
            status.update(label="Done!", state="complete")
            
            st.header('Key Points', divider='rainbow')
            st.write(key_points)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
else:
    st.warning('👈 Upload a PDF file to get started!')