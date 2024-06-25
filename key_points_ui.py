# run before running
# pip install transformers langchain pymupdf transformers torch streamlit sentencepiece
# streamlit run /Users/anishananda/Desktop/TCS intern/key_point_generator/key_points_ui.py

# python -m streamlit run key_points_ui.py 
# streamlit run key_points_ui.py

# importing
import streamlit as st
import time

# for the AI part
from transformers import pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import torch


# setting up the LLM
@st.cache_resource

# reading pdf
def read_pdf(file):#(file_path):
    #pdf_document = fitz.open(file_path)
    pdf_document = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text

# pegasus - apache
summarizer = pipeline("summarization", model="google/pegasus-cnn_dailymail")

# LLM based : MIT license
model_name = "EleutherAI/gpt-neo-125M"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPTNeoForCausalLM.from_pretrained(model_name)
model = model.half()  # Convert to half precision # to make faster

# preprocess through pegasus
def thru_pegasus(document):

    splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=50)
    split_documents = splitter.split_text(document)

    summaries = []
    for i, chunk in enumerate(split_documents):
        try:
            summary = summary = summarizer(chunk, max_length=500, min_length=40, do_sample=False)
            summaries.append(summary)
        except IndexError as e:
            print(f"IndexError at chunk {i}: {e}")
            summaries.append("Error in summarization")

    final_sum = ''
    for i in range(len(summaries)):
        if summaries[i] != "Error in summarization":
            final_sum += summaries[i][0]['summary_text'] +" "

    return final_sum

tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

# put through gpt neo
def extract_key_points(prompt, document, model, tokenizer, max_length=100000):
    # Create a prompt for the model to generate key points
    prompt = prompt + f" from the following document:\n{document}\n"

    # Tokenize and encode the input text
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Generate the key points
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=max_length,
        num_beams=5,
        early_stopping=True,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.eos_token_id  # Ensuring the pad token is correctly set
    )

    # Decode the generated text
    key_points = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the key points part from the response
    key_points_start = key_points.find("Key points:") + len("Key points:")
    key_points = key_points[key_points_start:].strip()

    return key_points




# Streamlit integration

# setting up page
st.set_page_config(page_title='Key Points Generator', page_icon='🏗️')
#st.experimental_rerun()

st.title('Key Points Generator')

with st.expander('About this app'):
  st.markdown('**What can this app do?**')
  st.info('This app helps users extract key points from a given document.')

  st.markdown('**How to use the app?**')
  st.warning('To engage with the app, go to the sidebar and upload a file in PDF format.')



# Sidebar for accepting input parameters
with st.sidebar:
    # Load data
    st.header('Input data')

    #st.markdown('**1. Use custom data**')
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_file is not None:
        st.success("File Upload Successful")
        #st.success("Summarizing...")


    #st.header('2. Set Length')
    #st.slider

sleep_time = 1

# Initiate the model building process
response = None
if uploaded_file : 
    with st.status("Running...", expanded=True) as status:
        time.sleep(sleep_time) 

        # Display data info
        st.header('Data Recieved', divider='rainbow')

        # loading
        try :
            status.update(label="Loading data...", state="running")
            time.sleep(sleep_time) 

            document = read_pdf(uploaded_file)

            status.update(label="Processing data...", state="running")
            time.sleep(sleep_time) 

            # put through pegasus
            #F_summary = thru_pegasus(document)

             # generate summary
            status.update(label="Summarizing...", state="running")
            time.sleep(sleep_time) 

            prompt = "Extract the key points"
            response = extract_key_points(prompt, document, model, tokenizer) #F_summary

        except FileNotFoundError :
            st.header("Error. Please Re-upload.")

        status.update(label="Process Completed!", state="complete", expanded=False)
            
        
if response:       
    st.header('Key Points', divider='rainbow')
    st.write(response)

else: # Ask for PDF upload if none is detected
    st.warning('👈 Upload a PDF file to get started!')


