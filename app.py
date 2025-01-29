import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
import openai

from dotenv import load_dotenv
load_dotenv()
## load the GROQ API Key
#os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")
os.environ['HUGGINGFACE_API_KEY']=os.getenv("HUGGINGFACE_API_KEY")

groq_api_key=os.getenv("GROQ_API_KEY")
huggingface_api_key=os.getenv("HUGGINGFACE_API_KEY")

llm=ChatGroq(groq_api_key=groq_api_key,model_name="llama-3.3-70b-versatile")

prompt=ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate respone based on the question
    <context>
    {context}
    <context>
    Question:{input}

    """

)

import tempfile

def process_uploaded_pdfs(uploaded_files):
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        all_documents = []
        
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.read())  # Write the uploaded file's content to the temp file
                temp_file.flush()
                loader = PyPDFLoader(temp_file.name)  # Use the temp file path with PyPDFLoader
                documents = loader.load()
                all_documents.extend(documents)

        # Split the documents into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(all_documents)

        # Create the FAISS vector store
        st.session_state.vectors = FAISS.from_documents(final_documents, st.session_state.embeddings)
        st.session_state.final_documents = final_documents

st.title("RAG Document Q&A With Groq And Lama3")

# File uploader for PDFs
uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if st.button("Process PDFs"):
    if uploaded_files:
        process_uploaded_pdfs(uploaded_files)
        st.write("Vector Database is ready")
    else:
        st.warning("Please upload at least one PDF file.")

user_prompt=st.text_input("Enter your query from the documents you provided")


#import time

if user_prompt:
    if "vectors" in st.session_state:
        document_chain=create_stuff_documents_chain(llm,prompt)
        retriever=st.session_state.vectors.as_retriever()
        retrieval_chain=create_retrieval_chain(retriever,document_chain)

        #start=time.process_time()
        response=retrieval_chain.invoke({'input':user_prompt})
        #print(f"Response time :{time.process_time()-start}")

        st.write(response['answer'])

        ## With a streamlit expander
        with st.expander("Document similarity Search"):
            for i,doc in enumerate(response['context']):
                st.write(doc.page_content)
                st.write('------------------------')
    else:
        st.warning("Please process the uploaded PDFs first.")





