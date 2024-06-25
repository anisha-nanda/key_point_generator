import streamlit as st
import time

# for the AI part
from transformers import pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import torch


# Page title
st.set_page_config(page_title='Key Points Generator', page_icon='🏗️')
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
if uploaded_file : 
    with st.status("Running ...", expanded=True) as status:
    
        st.write("Loading data ...")
        time.sleep(sleep_time) 

        st.write("Preparing data ...")
        time.sleep(sleep_time)

        # put through pegasus

            
    
        
        
    

    # Display data info
    st.header('Input data', divider='rainbow')
    
    
# Ask for CSV upload if none is detected
else:
    st.warning('👈 Upload a PDF file to get started!')


