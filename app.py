import streamlit as st
import PyPDF2
from haystack.dataclasses import Document

from haystack_rag import rag_pipeline

##### Streamlit APP
st.title("'AI Resume Analyst'")
info = '''How to pick a perfect candidate for your job?


Easy!

Upload candidate resumes, formulate your questions clear and specific and let AI help you with finding the perfect candidate for your team.
'''
st.markdown(info)


st.sidebar.title("Your OpenAI API key here")
openai_api_key = st.sidebar.text_input('OpenAI API key')
st.sidebar.divider()
st.sidebar.title("Candidates resume")


with st.form('my_form'):
    query = st.text_area('Enter the question about candidates')
    submitted = st.form_submit_button('Answer')
#   if not openai_api_key.startswith('sk-'):
#     st.warning('Please enter your OpenAI API key!', icon='⚠')
  
  #if submitted and openai_api_key.startswith('sk-'): 
  #if submitted:

with st.form(key='job description_form'):
    job_description = st.text_area('Enter the job description')
    submitted = st.form_submit_button(label='Just Enter',disabled=True)

uploaded_file = st.sidebar.file_uploader(label="CV",label_visibility="hidden")
if uploaded_file:
    # Exstracting the text from PDF
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ''
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()

#document that will be splitted, embedded and written in the document store
try:
    document = [Document(content=text)]  
except Exception as e:
    st.sidebar.error("Please enter a candidate CV")

if submitted:
    result = rag_pipeline(document,query)
    st.info(result['generator']['replies'][0])